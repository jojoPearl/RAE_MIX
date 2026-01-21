import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, time, datetime, glob, re, json, argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

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

# ----------------------------
# Loader
# ----------------------------
def load_and_transform(path: str, target_size: int = 448) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0)
    return tensor.to(device=DEVICE, dtype=DTYPE)

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

def lora_trainable_params(dit: nn.Module):
    return [p for p in dit.parameters() if p.requires_grad]

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
    - 2D positional encoding keeps spatial alignment
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
        pos = torch.stack([xx, yy], dim=-1).view(1, grid * grid, 2)  # [1,L,2]
        return pos

    def forward(self, ctrl_img: torch.Tensor) -> torch.Tensor:
        x = self.stem(ctrl_img)
        x = self.proj(x)  # [B, Denc, H, W]
        grid = int(self.token_len ** 0.5)

        if self.use_max_pool:
            x = F.adaptive_max_pool2d(x, (grid, grid))
        else:
            x = F.adaptive_avg_pool2d(x, (grid, grid))

        x = x.flatten(2).transpose(1,2)  # [B,L,Denc]
        pos = self._pos_grid(grid, device=x.device, dtype=x.dtype)  # [1,L,2]
        x = x + self.pos_proj(pos)
        return x * self.log_scale.exp()

# ----------------------------
# Linear path training (velocity)
# ----------------------------
def sample_linear_path(z0: torch.Tensor, t: torch.Tensor):
    B = z0.shape[0]
    z1 = torch.randn_like(z0)
    t_view = t.view(B,1,1,1)
    z_t = (1.0 - t_view) * z0 + t_view * z1
    v_true = z1 - z0
    return z_t, v_true

# ----------------------------
# Build s_content in encoder space (BEFORE s_projector)
# ----------------------------
def build_s_content_enc(dit, z_t, t, y):
    t_emb = dit.t_embedder(t)        # [B, D_enc]
    y_emb = dit.y_embedder(y, False) # [B, D_enc]
    c = F.silu(t_emb + y_emb)

    s = dit.s_embedder(z_t)          # [B, L, D_enc]
    if getattr(dit, "use_pos_embed", False) and hasattr(dit, "pos_embed"):
        s = s + _interp_pos_embed_if_needed(dit.pos_embed, s.shape[1])

    for i in range(dit.num_encoder_blocks):
        s = dit.blocks[i](s, c, feat_rope=dit.enc_feat_rope)

    s = F.silu(t_emb.unsqueeze(1) + s)
    return s

# ----------------------------
# Configs
# ----------------------------
@dataclass
class TrainCfg:
    image_size: int = 448
    steps: int = 3000
    lr_adapter: float = 1e-4
    lr_lora: float = 1e-4
    weight_decay: float = 0.0
    log_every: int = 50
    save_every: int = 500
    control_weight: float = 1.0
    grad_clip: float = 1.0
    seed: int = 0
    smooth_w: float = 0.02

@dataclass
class InferCfg:
    steps: int = 64
    control_weight: float = 0.6
    gamma_c: float = 1.2
    img2img_strength: float = 0.6

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
# Train: canny OR depth OR both (two adapters)
# ----------------------------
def train_one(
    base_image_path: str,
    canny_path: str,
    depth_path: str,
    out_train_dir: str,
    class_id: int,
    ctrl_mode: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    cfg: TrainCfg,
):
    ensure_dir(out_train_dir)
    torch.manual_seed(cfg.seed)

    manager = ModelManager(device=device)
    rae = manager.load_rae().eval()
    dit = manager.load_dit()
    if dit is None:
        raise RuntimeError("Failed to load DiT/DDT")
    dit = dit.to(device=device, dtype=dtype)

    for p in rae.parameters(): p.requires_grad_(False)
    for p in dit.parameters(): p.requires_grad_(False)

    replaced = []
    if use_lora:
        replaced = inject_lora(dit, r=lora_r, alpha=lora_alpha, dropout=0.0)
        dit = dit.to(device=device, dtype=dtype)
        dit.train()
        print(f"[LoRA] injected {len(replaced)} layers.")
    else:
        dit.eval()

    d_enc = int(getattr(dit, "encoder_hidden_size", 1152))
    token_len = int(getattr(dit.s_embedder, "num_patches", 1024))
    if hasattr(dit, "s_projector") and isinstance(dit.s_projector, nn.Linear):
        d_enc = int(dit.s_projector.in_features)

    ctrl_imgs = load_ctrl_imgs(ctrl_mode, canny_path, depth_path, cfg.image_size)
    adapters = {}
    for k in ctrl_imgs.keys():
        use_max_pool = True if k == "canny" else False
        adapters[k] = ControlAdapter(d_enc=d_enc, token_len=token_len, use_max_pool=use_max_pool).to(device=device, dtype=dtype).train()

    params = []
    for k, ad in adapters.items():
        params.append({"params": ad.parameters(), "lr": cfg.lr_adapter})
    if use_lora:
        params.append({"params": lora_trainable_params(dit), "lr": cfg.lr_lora})

    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    base_img = load_and_transform(base_image_path, cfg.image_size)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
        z0 = rae.encode(base_img)
    y = torch.tensor([class_id], device=device, dtype=torch.long)

    with torch.no_grad():
        recon = rae.decode(z0).float().clamp(0,1)
        save_image(recon, os.path.join(out_train_dir, "recon_preview.png"))

    meta = {
        "base_image_path": base_image_path,
        "canny_path": canny_path,
        "depth_path": depth_path,
        "class_id": class_id,
        "ctrl_mode": ctrl_mode,
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_cfg": cfg.__dict__,
        "time": now_ts(),
        "replaced": replaced,
    }
    write_json(meta, os.path.join(out_train_dir, "run_meta.json"))

    print("=== TRAIN START ===")
    print(f"out_train_dir: {out_train_dir}")
    print(f"ctrl_mode={ctrl_mode} use_lora={use_lora}")
    print(f"latent={tuple(z0.shape)} token_len={token_len} d_enc={d_enc}")

    t0 = time.time()
    ema = None

    for step in range(1, cfg.steps+1):
        t = torch.rand((1,), device=device, dtype=dtype).clamp(1e-4, 1-1e-4)
        z_t, v_true = sample_linear_path(z0, t)

        if use_lora:
            s_content = build_s_content_enc(dit, z_t, t, y)
        else:
            with torch.no_grad():
                s_content = build_s_content_enc(dit, z_t, t, y)

        s_ctrl_sum = 0.0
        smooth_loss = 0.0

        for k, img in ctrl_imgs.items():
            ad = adapters[k]
            s_ctrl = ad(img)

            sc = s_content.std(dim=(1,2), keepdim=True).detach()
            cc = s_ctrl.std(dim=(1,2), keepdim=True).detach()
            s_ctrl = s_ctrl * (sc / (cc + 1e-6))

            s_ctrl_sum = s_ctrl_sum + s_ctrl
            smooth_loss = smooth_loss + (s_ctrl[:,1:] - s_ctrl[:,:-1]).pow(2).mean()

        if len(ctrl_imgs) > 1:
            s_ctrl_sum = s_ctrl_sum / float(len(ctrl_imgs))
            smooth_loss = smooth_loss / float(len(ctrl_imgs))

        s_final = s_content + cfg.control_weight * s_ctrl_sum

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
            v_pred = dit(z_t, t, y, s=s_final, mask=None)
            loss_main = F.mse_loss(v_pred.float(), v_true.float())
            loss = loss_main + cfg.smooth_w * smooth_loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        if cfg.grad_clip and cfg.grad_clip > 0:
            for k, ad in adapters.items():
                torch.nn.utils.clip_grad_norm_(ad.parameters(), cfg.grad_clip)
            if use_lora:
                torch.nn.utils.clip_grad_norm_(lora_trainable_params(dit), cfg.grad_clip)

        scaler.step(opt)
        scaler.update()

        lv = float(loss.detach().cpu())
        ema = lv if ema is None else (0.95*ema + 0.05*lv)

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            print(f"[{step:05d}/{cfg.steps}] loss={lv:.6f} ema={ema:.6f} time={dt:.1f}s")

        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt_path = os.path.join(out_train_dir, f"ckpt_step_{step}.pt")

            payload = {
                "step": step,
                "ctrl_mode": ctrl_mode,
                "class_id": class_id,
                "use_lora": use_lora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "adapters": {k: ad.state_dict() for k, ad in adapters.items()},
            }
            if use_lora:
                payload["lora"] = extract_lora_state_dict(dit)
                payload["replaced"] = replaced

            torch.save(payload, ckpt_path)
            torch.save(payload, os.path.join(out_train_dir, "latest.pt"))
            print(f"Saved: {ckpt_path}")

    print("=== TRAIN DONE ===")

# ----------------------------
# Infer sampler (Control-CFG), txt2img or img2img init
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
    img2img_strength: float,
    out_path: str,
):
    B = 1
    latent_shape = (B, 768, 32, 32)

    if infer_mode == "txt2img":
        z = torch.randn(latent_shape, device=device, dtype=dtype)
    elif infer_mode == "img2img":
        assert base_z0 is not None, "img2img requires base_z0"
        eps = torch.randn_like(base_z0)
        s = float(img2img_strength)
        z = (1.0 - s) * base_z0 + s * eps
    else:
        raise ValueError("infer_mode must be txt2img or img2img")

    # precompute ctrl embeddings (fixed)
    s_ctrl_base = 0.0
    for k, img in ctrl_imgs.items():
        s_ctrl_base = s_ctrl_base + adapters[k](img)
    if len(ctrl_imgs) > 1:
        s_ctrl_base = s_ctrl_base / float(len(ctrl_imgs))

    ts = torch.linspace(1.0, 0.0, steps+1, device=device, dtype=dtype)

    for i in range(steps):
        t_curr = ts[i].repeat(B)
        t_next = ts[i+1].repeat(B)
        dt = (t_curr - t_next).view(B,1,1,1)

        s_content = build_s_content_enc(dit, z, t_curr, y)

        sc = s_content.std(dim=(1,2), keepdim=True)
        cc = s_ctrl_base.std(dim=(1,2), keepdim=True)
        s_ctrl = s_ctrl_base * (sc/(cc + 1e-6))

        s_on  = s_content + control_weight * s_ctrl
        v_on  = dit(z, t_curr, y, s=s_on, mask=None)
        v_off = dit(z, t_curr, y, s=s_content, mask=None)

        # DEBUG: check control effect
        if (i % 10) == 0:
            delta = (v_on - v_off).abs().mean().item()
            print(f"[DEBUG] i={i:03d} delta_v={delta:.6e}")

        v = v_off + gamma_c * (v_on - v_off)

        z = z - v * dt
        if torch.isnan(z).any():
            z = torch.nan_to_num(z)

    img = rae.decode(z).float().clamp(0,1)
    save_image(img, out_path)
    print("Saved:", out_path)

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
# main entry
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train_infer", choices=["train", "infer", "train_infer"])
    ap.add_argument("--ctrl_mode", type=str, default="canny", choices=["canny","depth","both"])
    ap.add_argument("--infer_mode", type=str, default="img2img", choices=["img2img","txt2img"])

    ap.add_argument("--base_image", type=str, default="assets/out_ctrl/r.png")
    ap.add_argument("--canny", type=str, default="assets/out_ctrl/r_canny.png")
    ap.add_argument("--depth", type=str, default="assets/out_ctrl/r_depth.png")

    ap.add_argument("--class_id", type=int, default=207)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)

    # train
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--lr_adapter", type=float, default=1e-4)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--control_weight", type=float, default=1.0)
    ap.add_argument("--smooth_w", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)

    # infer
    ap.add_argument("--infer_steps", type=int, default=64)
    ap.add_argument("--infer_control_weight", type=float, default=0.6)
    ap.add_argument("--gamma_c", type=float, default=1.2)
    ap.add_argument("--img2img_strength", type=float, default=0.6)

    args = ap.parse_args()

    out_train_dir = ensure_dir(os.path.join(OUT_ROOT, "train", "adapter", args.ctrl_mode))
    out_infer_dir = ensure_dir(os.path.join(OUT_ROOT, "infer", args.ctrl_mode, args.infer_mode))

    if not os.path.exists(args.base_image):
        raise SystemExit(f"Base image not found: {args.base_image}")
    if not os.path.exists(args.canny):
        raise SystemExit(f"Canny image not found: {args.canny}")
    if not os.path.exists(args.depth):
        raise SystemExit(f"Depth image not found: {args.depth}")

    # TRAIN
    if args.mode in ("train", "train_infer"):
        train_cfg = TrainCfg(
            image_size=448,
            steps=args.steps,
            lr_adapter=args.lr_adapter,
            lr_lora=args.lr_lora,
            weight_decay=0.0,
            log_every=args.log_every,
            save_every=args.save_every,
            control_weight=args.control_weight,
            grad_clip=1.0,
            seed=args.seed,
            smooth_w=args.smooth_w,
        )
        train_one(
            base_image_path=args.base_image,
            canny_path=args.canny,
            depth_path=args.depth,
            out_train_dir=out_train_dir,
            class_id=args.class_id,
            ctrl_mode=args.ctrl_mode,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            cfg=train_cfg,
        )

    # INFER
    if args.mode in ("infer", "train_infer"):
        manager = ModelManager(device=device)
        rae = manager.load_rae().eval()
        dit = manager.load_dit()
        if dit is None:
            raise RuntimeError("Failed to load DiT/DDT")
        dit = dit.to(device=device, dtype=dtype).eval()
        for p in rae.parameters(): p.requires_grad_(False)
        for p in dit.parameters(): p.requires_grad_(False)

        _ = inject_lora(dit, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0)

        ckpt_path = os.path.join(out_train_dir, "latest.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = latest_ckpt(out_train_dir, pattern="ckpt_step_*.pt")
        if ckpt_path is None or (not os.path.exists(ckpt_path)):
            raise SystemExit(f"No ckpt found in {out_train_dir}. Train first.")

        adapters, ck = load_ckpt_and_build_adapters(dit, ckpt_path)

        # ---- DEBUG + SAFETY: ckpt must match args.ctrl_mode ----
        print("[DEBUG] loaded ckpt_path:", ckpt_path)
        print("[DEBUG] ckpt ctrl_mode:", ck.get("ctrl_mode"))
        print("[DEBUG] args ctrl_mode:", args.ctrl_mode)
        print("[DEBUG] adapter keys:", list(adapters.keys()))
        if ck.get("ctrl_mode") != args.ctrl_mode:
            raise RuntimeError(
                f"CTRL_MODE mismatch: ckpt={ck.get('ctrl_mode')} but args={args.ctrl_mode}. "
                f"Check your folder: {out_train_dir}"
            )

        # ---- IMPORTANT: load control images by ckpt mode (robust) ----
        ctrl_imgs = load_ctrl_imgs(ck["ctrl_mode"], args.canny, args.depth, 448)

        base_z0 = None
        if args.infer_mode == "img2img":
            base_img = load_and_transform(args.base_image, 448)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE_MODERN):
                base_z0 = rae.encode(base_img).to(device=device, dtype=dtype)

        ts = now_ts()
        out_path = os.path.join(out_infer_dir, f"gen_{args.class_id}_{args.ctrl_mode}_{args.infer_mode}_{ts}.png")

        infer_cfg = InferCfg(
            steps=args.infer_steps,
            control_weight=args.infer_control_weight,
            gamma_c=args.gamma_c,
            img2img_strength=args.img2img_strength,
        )

        y = torch.tensor([args.class_id], device=device, dtype=torch.long)
        ode_sample_with_control_cfg(
            dit=dit, rae=rae,
            adapters=adapters,
            ctrl_imgs=ctrl_imgs,
            y=y,
            infer_mode=args.infer_mode,
            base_z0=base_z0,
            steps=infer_cfg.steps,
            control_weight=infer_cfg.control_weight,
            gamma_c=infer_cfg.gamma_c,
            img2img_strength=infer_cfg.img2img_strength,
            out_path=out_path,
        )

if __name__ == "__main__":
    main()
