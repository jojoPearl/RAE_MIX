import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, time, datetime, glob, re, json, argparse, random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

# ----------------------------
# Path Setup (project root)
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# ----------------------------
# Project Imports
# ----------------------------
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE, DTYPE_MODERN

device = DEVICE
dtype  = DTYPE

# ----------------------------
# I/O + dirs
# ----------------------------
OUT_ROOT = "assets/out_ctrl"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# ----------------------------
# Loader
# ----------------------------
# Input is resized to 448x448 so that RAE produces a 32x32 latent map,
# matching the fixed DiT token grid (1024 tokens) and control alignment.
def load_and_transform(path: str, target_size: int = 448) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0)
    return tensor.to(device=DEVICE, dtype=DTYPE)

# Normalize input to [B,3,H,W].
def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] == 3: return x
    if x.shape[1] == 1: return x.repeat(1,3,1,1)
    return x[:, :3]

def _interp_pos_embed_if_needed(pos_embed: torch.Tensor, target_len: int) -> torch.Tensor:
    if pos_embed.shape[1] == target_len:
        return pos_embed
    src_len = pos_embed.shape[1]
    src_grid = int(src_len ** 0.5)
    tgt_grid = int(target_len ** 0.5)

    # safe fallback (truncate/pad)
    if src_grid * src_grid != src_len or tgt_grid * tgt_grid != target_len:
        if target_len < src_len:
            return pos_embed[:, :target_len]
        pad = torch.zeros((1, target_len - src_len, pos_embed.shape[-1]),
                          device=pos_embed.device, dtype=pos_embed.dtype)
        return torch.cat([pos_embed, pad], dim=1)

    pe = pos_embed.transpose(1,2).reshape(1, -1, src_grid, src_grid)
    pe = F.interpolate(pe, size=(tgt_grid, tgt_grid), mode="bicubic", align_corners=False)
    return pe.flatten(2).transpose(1,2)

# ----------------------------
# Minimal LoRA (optional)
# ----------------------------
class LoRALinear(nn.Module):
    """
    y = Wx + (B(Ax)) * (alpha/r)
    """
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dev = base.weight.device
        dt  = base.weight.dtype
        self.A = nn.Linear(base.in_features, r, bias=False).to(device=dev, dtype=dt)
        self.B = nn.Linear(r, base.out_features, bias=False).to(device=dev, dtype=dt)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.drop(x))) * self.scale

def inject_lora(
    dit: nn.Module,
    r=8,
    alpha=16,
    dropout=0.0,
    name_keywords=("qkv","q_proj","k_proj","v_proj","out_proj")
):
    replaced = []

    def _rec(m: nn.Module, prefix=""):
        for n, child in list(m.named_children()):
            full = f"{prefix}.{n}" if prefix else n
            if isinstance(child, nn.Linear) and any(k in n for k in name_keywords):
                setattr(m, n, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(full)
            else:
                _rec(child, full)

    _rec(dit)
    return replaced

def extract_lora_state_dict(dit: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for k, v in dit.state_dict().items():
        if (".A." in k) or (".B." in k):
            sd[k] = v.detach().cpu()
    return sd

def load_lora_state_dict(dit: nn.Module, lora_sd: Dict[str, torch.Tensor]):
    sd = dit.state_dict()
    for k, v in lora_sd.items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))
    dit.load_state_dict(sd, strict=True)

# ----------------------------
# Control Adapter -> encoder-dim tokens
# ----------------------------
class ControlAdapter(nn.Module):
    """
    ctrl_img -> s_ctrl_enc [B,L,D_enc]
    - max pool helps sparse edges (canny)
    - positional encoding keeps spatial alignment
    """
    def __init__(self, d_enc: int, token_len: int, use_max_pool=True):
        super().__init__()
        self.d_enc = d_enc
        self.token_len = token_len
        self.use_max_pool = use_max_pool

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.proj = nn.Conv2d(128, d_enc, 1)
        self.pos_proj = nn.Linear(2, d_enc, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(-4.0))  # exp(-4)=0.018

    def _pos_grid(self, grid: int, device, dtype):
        coords = torch.linspace(-1.0, 1.0, grid, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        pos = torch.stack([xx, yy], dim=-1).view(1, grid * grid, 2)
        return pos

    def forward(self, ctrl_img: torch.Tensor) -> torch.Tensor:
        x = self.stem(ctrl_img)
        x = self.proj(x)
        grid = int(self.token_len ** 0.5)

        if self.use_max_pool:
            x = F.adaptive_max_pool2d(x, (grid, grid))
        else:
            x = F.adaptive_avg_pool2d(x, (grid, grid))

        x = x.flatten(2).transpose(1,2)
        pos = self._pos_grid(grid, device=x.device, dtype=x.dtype)
        x = x + self.pos_proj(pos)
        return x * self.log_scale.exp()

# ----------------------------
# Build s_content in encoder space
# ----------------------------
def build_s_content_enc(dit, z_t, t, y):
    t_emb = dit.t_embedder(t)
    y_emb = dit.y_embedder(y, False)
    c = F.silu(t_emb + y_emb)

    s = dit.s_embedder(z_t)
    if getattr(dit, "use_pos_embed", False) and hasattr(dit, "pos_embed"):
        s = s + _interp_pos_embed_if_needed(dit.pos_embed, s.shape[1])

    for i in range(dit.num_encoder_blocks):
        s = dit.blocks[i](s, c, feat_rope=dit.enc_feat_rope)

    s = F.silu(t_emb.unsqueeze(1) + s)
    return s

# ----------------------------
# checkpoint utils
# ----------------------------
def latest_ckpt(dir_path: str, pattern="ckpt_step_*.pt"):
    ckpts = glob.glob(os.path.join(dir_path, pattern))
    if not ckpts:
        return None
    def step_num(p):
        m = re.search(r"(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=step_num)

# ----------------------------
# control image getter
# ----------------------------
def load_ctrl_imgs(ctrl_mode: str, canny_path: str, depth_path: str, size: int) -> Dict[str, torch.Tensor]:
    if ctrl_mode == "canny":
        return {"canny": _ensure_3ch(load_and_transform(canny_path, size))}
    if ctrl_mode == "depth":
        return {"depth": _ensure_3ch(load_and_transform(depth_path, size))}
    if ctrl_mode == "both":
        return {
            "canny": _ensure_3ch(load_and_transform(canny_path, size)),
            "depth": _ensure_3ch(load_and_transform(depth_path, size)),
        }
    raise ValueError(f"Unknown ctrl_mode={ctrl_mode}")

# ----------------------------
# load ckpt
# ----------------------------
def load_ckpt_and_build_adapters(dit, ckpt_path: str) -> Tuple[Dict[str, ControlAdapter], Dict]:
    ck = torch.load(ckpt_path, map_location="cpu")
    ctrl_mode = ck["ctrl_mode"]

    d_enc = int(getattr(dit, "encoder_hidden_size", 1152))
    token_len = int(getattr(dit.s_embedder, "num_patches", 1024))
    if hasattr(dit, "s_projector") and isinstance(dit.s_projector, nn.Linear):
        d_enc = int(dit.s_projector.in_features)

    if ctrl_mode == "canny":
        keys = ["canny"]
    elif ctrl_mode == "depth":
        keys = ["depth"]
    elif ctrl_mode == "both":
        keys = ["canny", "depth"]
    else:
        raise ValueError(f"Unknown ctrl_mode in ckpt: {ctrl_mode}")

    adapters = {}
    for k in keys:
        use_max_pool = True if k == "canny" else False
        ad = ControlAdapter(d_enc=d_enc, token_len=token_len, use_max_pool=use_max_pool).to(device=device, dtype=dtype).eval()
        ad.load_state_dict(ck["adapters"][k], strict=True)
        for p in ad.parameters(): p.requires_grad_(False)
        adapters[k] = ad

    if ck.get("use_lora", False) and "lora" in ck:
        load_lora_state_dict(dit, ck["lora"])
        print("[Info] Loaded LoRA weights from ckpt.")

    return adapters, ck

# ----------------------------
# Infer sampler (Control-CFG)
# - 支持 pure-noise：两种模式都从 t=1 的纯噪声开始
# ----------------------------
@torch.no_grad()
def ode_sample_with_control_cfg(
    dit, rae,
    adapters: Dict[str, ControlAdapter],
    ctrl_imgs: Dict[str, torch.Tensor],
    y: torch.Tensor,
    infer_mode: str,
    base_z0: Optional[torch.Tensor],
    steps: int,
    control_weight: float,
    gamma_c: float,
    noise_level: float,          # for img2img mix (if not pure_noise)
    pure_noise: bool,            # <-- 新增：强制纯噪声起步
    out_path: str,
):
    B = 1
    latent_shape = (B, 768, 32, 32)

    # 预计算 ctrl embeddings
    s_ctrl_base = 0.0
    for k, img in ctrl_imgs.items():
        s_ctrl_base = s_ctrl_base + adapters[k](img)
    if len(ctrl_imgs) > 1:
        s_ctrl_base = s_ctrl_base / float(len(ctrl_imgs))

    # 纯噪声：两种模式都一样，从 t=1 开始
    if pure_noise:
        z = torch.randn(latent_shape, device=device, dtype=dtype)
        t_start = 1.0
    else:
        # 原始逻辑（保留，以防你以后不用 pure_noise）
        s = float(noise_level)
        s = max(1e-4, min(1.0, s))

        if infer_mode == "txt2img":
            z = torch.randn(latent_shape, device=device, dtype=dtype)
            t_start = 1.0
        elif infer_mode == "img2img":
            assert base_z0 is not None, "img2img requires base_z0 when pure_noise=False"
            eps = torch.randn_like(base_z0)
            z = (1.0 - s) * base_z0 + s * eps
            t_start = s
        else:
            raise ValueError("infer_mode must be txt2img or img2img")

    # integrate from t_start -> 0
    ts = torch.linspace(t_start, 0.0, steps + 1, device=device, dtype=dtype)

    for i in range(steps):
        t_curr = ts[i].repeat(B)
        t_next = ts[i+1].repeat(B)
        dt = (t_curr - t_next).view(B,1,1,1)  # >0

        s_content = build_s_content_enc(dit, z, t_curr, y)

        sc = s_content.std(dim=(1,2), keepdim=True)
        cc = s_ctrl_base.std(dim=(1,2), keepdim=True)
        s_ctrl = s_ctrl_base * (sc/(cc + 1e-6))

        s_on  = s_content + control_weight * s_ctrl
        v_on  = dit(z, t_curr, y, s=s_on, mask=None)
        v_off = dit(z, t_curr, y, s=s_content, mask=None)

        v = v_off + gamma_c * (v_on - v_off)
        z = z - v * dt

        if torch.isnan(z).any():
            z = torch.nan_to_num(z)

    img = rae.decode(z).float().clamp(0,1)
    save_image(img, out_path)
    print("Saved:", out_path)

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="infer", choices=["train", "infer", "train_infer"])
    ap.add_argument("--ctrl_mode", type=str, default="canny", choices=["canny","depth","both"])
    ap.add_argument("--infer_mode", type=str, default="txt2img", choices=["img2img","txt2img"])

    ap.add_argument("--base_image", type=str, default="assets/out_ctrl/image2.png")
    ap.add_argument("--canny", type=str, default="assets/out_ctrl/r_canny.png")
    ap.add_argument("--depth", type=str, default="assets/out_ctrl/r_depth.png")

    ap.add_argument("--class_id", type=int, default=207)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)

    # infer
    ap.add_argument("--infer_steps", type=int, default=64)
    ap.add_argument("--infer_control_weight", type=float, default=0.6)
    ap.add_argument("--gamma_c", type=float, default=1.2)

    # 仍保留 img2img_strength（当 pure_noise=False 才有用）
    ap.add_argument("--img2img_strength", type=float, default=0.6)

    # 关键：强制纯噪声（两种 infer_mode 都生效）
    ap.add_argument("--pure_noise", action="store_true",
                    help="Force start from pure noise (t=1.0). Works for both txt2img and img2img.")

    ap.add_argument("--seed", type=int, default=0)

    # noise sweep（保留）
    ap.add_argument("--noise_sweep", action="store_true")
    ap.add_argument("--noise_min", type=float, default=0.2)
    ap.add_argument("--noise_max", type=float, default=0.8)
    ap.add_argument("--noise_num", type=int, default=5)

    args = ap.parse_args()

    set_seed(args.seed)

    out_train_dir = ensure_dir(os.path.join(OUT_ROOT, "train", "adapter", args.ctrl_mode))
    out_infer_dir = ensure_dir(os.path.join(OUT_ROOT, "infer", args.ctrl_mode, args.infer_mode))

    # ---- 文件检查：按需检查 ----
    if args.ctrl_mode in ("canny", "both") and (not os.path.exists(args.canny)):
        raise SystemExit(f"Canny image not found: {args.canny}")
    if args.ctrl_mode in ("depth", "both") and (not os.path.exists(args.depth)):
        raise SystemExit(f"Depth image not found: {args.depth}")

    # base_image 只在 img2img 且 pure_noise=False 时才需要
    if (args.infer_mode == "img2img") and (not args.pure_noise):
        if not os.path.exists(args.base_image):
            raise SystemExit(f"Base image not found: {args.base_image}")

    if args.mode not in ("infer", "train_infer"):
        raise SystemExit("This final version is intended for inference. Use --mode infer.")

    # load models
    manager = ModelManager(device=device)
    rae = manager.load_rae().eval()
    dit = manager.load_dit()
    if dit is None:
        raise RuntimeError("Failed to load DiT/DDT")
    dit = dit.to(device=device, dtype=dtype).eval()
    for p in rae.parameters(): p.requires_grad_(False)
    for p in dit.parameters(): p.requires_grad_(False)

    # lora injection (safe even if not used)
    replaced = inject_lora(dit, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0)
    if len(replaced) > 0:
        print(f"[LoRA] injected modules exist: {len(replaced)} (weights loaded only if ckpt contains them)")

    # load ckpt
    ckpt_path = os.path.join(out_train_dir, "latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = latest_ckpt(out_train_dir, pattern="ckpt_step_*.pt")
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise SystemExit(f"No ckpt found in {out_train_dir}. Train first.")

    adapters, ck = load_ckpt_and_build_adapters(dit, ckpt_path)

    # ctrl imgs
    ctrl_imgs = load_ctrl_imgs(args.ctrl_mode, args.canny, args.depth, 448)

    # base_z0 for img2img (only if needed)
    base_z0 = None
    if args.infer_mode == "img2img" and (not args.pure_noise):
        base_img = load_and_transform(args.base_image, 448)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
            base_z0 = rae.encode(base_img).to(device=device, dtype=dtype)

    # decide noise levels (只有 pure_noise=False 的 img2img 才有意义；这里保留兼容)
    if args.noise_sweep:
        if args.noise_num <= 1:
            noise_levels = [float(args.noise_min)]
        else:
            noise_levels = torch.linspace(args.noise_min, args.noise_max, args.noise_num).tolist()
    else:
        noise_levels = [float(args.img2img_strength)]

    ts = now_ts()
    y = torch.tensor([args.class_id], device=device, dtype=torch.long)

    for nl in noise_levels:
        tag = "pureNoise" if args.pure_noise else f"noise{nl:.3f}"
        out_path = os.path.join(
            out_infer_dir,
            f"gen_{args.class_id}_{args.ctrl_mode}_{args.infer_mode}_{tag}_{ts}.png"
        )

        ode_sample_with_control_cfg(
            dit=dit, rae=rae,
            adapters=adapters,
            ctrl_imgs=ctrl_imgs,
            y=y,
            infer_mode=args.infer_mode,
            base_z0=base_z0,
            steps=args.infer_steps,
            control_weight=args.infer_control_weight,
            gamma_c=args.gamma_c,
            noise_level=nl,
            pure_noise=args.pure_noise,
            out_path=out_path,
        )

if __name__ == "__main__":
    main()

