import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, datetime, glob, re, json, argparse, random
from typing import Dict, Optional, Tuple

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

OUT_ROOT = "assets/out_ctrl"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
# Control Adapter
# ----------------------------
class ControlAdapter(nn.Module):
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

        # 你原来是 -4.0 (exp=0.018) 太弱，这里保留参数，但我们会额外乘 ctrl_gain 强化
        self.log_scale = nn.Parameter(torch.tensor(-4.0))

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


def latest_ckpt(dir_path: str, pattern="ckpt_step_*.pt"):
    ckpts = glob.glob(os.path.join(dir_path, pattern))
    if not ckpts:
        return None
    def step_num(p):
        m = re.search(r"(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=step_num)


def load_ckpt_and_build_adapters(dit, ckpt_path: str) -> Tuple[Dict[str, ControlAdapter], Dict]:
    ck = torch.load(ckpt_path, map_location="cpu")
    ctrl_mode = ck["ctrl_mode"]
    if ctrl_mode != "canny":
        raise ValueError(f"This txt2img script expects canny ckpt, but got ctrl_mode={ctrl_mode}")

    d_enc = int(getattr(dit, "encoder_hidden_size", 1152))
    token_len = int(getattr(dit.s_embedder, "num_patches", 1024))
    if hasattr(dit, "s_projector") and isinstance(dit.s_projector, nn.Linear):
        d_enc = int(dit.s_projector.in_features)

    ad = ControlAdapter(d_enc=d_enc, token_len=token_len, use_max_pool=True).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ck["adapters"]["canny"], strict=True)
    for p in ad.parameters():
        p.requires_grad_(False)

    return {"canny": ad}, ck


# ----------------------------
# Stronger control injection: per-block (ControlNet-ish)
# ----------------------------
def build_s_content_enc_with_ctrl(
    dit,
    z_t: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    s_ctrl: Optional[torch.Tensor],
    ctrl_inject: str = "per_block",
    ctrl_block_scale: float = 1.0,
):
    """
    ctrl_inject:
      - "none": no control
      - "once": only add before blocks
      - "per_block": add before each encoder block (stronger)
    """
    t_emb = dit.t_embedder(t)
    y_emb = dit.y_embedder(y, False)
    c = F.silu(t_emb + y_emb)

    s = dit.s_embedder(z_t)
    if getattr(dit, "use_pos_embed", False) and hasattr(dit, "pos_embed"):
        s = s + _interp_pos_embed_if_needed(dit.pos_embed, s.shape[1])

    if (s_ctrl is not None) and (ctrl_inject == "once"):
        s = s + ctrl_block_scale * s_ctrl

    for i in range(dit.num_encoder_blocks):
        if (s_ctrl is not None) and (ctrl_inject == "per_block"):
            s = s + ctrl_block_scale * s_ctrl
        s = dit.blocks[i](s, c, feat_rope=dit.enc_feat_rope)

    s = F.silu(t_emb.unsqueeze(1) + s)
    return s


# ----------------------------
# Optional: CLIP semantic guidance
# ----------------------------
def _load_clip():
    """
    兼容两种来源：
      1) open_clip（如果你装了）
      2) torchvision 的 clip（如果版本支持）
    """
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(device=device).eval()
        return ("open_clip", model, tokenizer, preprocess)
    except Exception:
        pass

    try:
        # torchvision>=0.14 有些环境有 CLIP，但接口不一定一致，这里尽量稳妥
        from torchvision.models import clip as tv_clip
        model = tv_clip.clip_vit_b32(pretrained=True).to(device=device).eval()
        return ("torchvision", model, None, None)
    except Exception as e:
        raise RuntimeError(
            "CLIP not available. Install open_clip:\n"
            "  pip install open_clip_torch\n"
            "or use an environment with torchvision CLIP."
        ) from e


@torch.enable_grad()
def clip_guidance_grad(
    clip_pack,
    img_01: torch.Tensor,   # [1,3,H,W] in 0..1
    prompt: str,
) -> torch.Tensor:
    """
    返回一个标量 loss（越小越符合 prompt）。
    我们做：maximize cosine similarity => minimize negative similarity
    """
    kind, model, tokenizer, preprocess = clip_pack

    # resize for CLIP
    img = F.interpolate(img_01, size=(224, 224), mode="bilinear", align_corners=False)
    img = img.clamp(0, 1)

    # normalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device).view(1,3,1,1)
    img_n = (img - mean) / std

    if kind == "open_clip":
        text = tokenizer([prompt]).to(device=img.device)
        img_feat = model.encode_image(img_n)
        txt_feat = model.encode_text(text)
    else:
        # torchvision clip 兼容性不稳定：如果你这条路报错，就安装 open_clip
        raise RuntimeError("torchvision CLIP guidance path not supported reliably. Please install open_clip_torch.")

    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-6)

    sim = (img_feat * txt_feat).sum(dim=-1)  # [1]
    loss = -sim.mean()
    return loss


# ----------------------------
# txt2img sampler (Control-CFG + optional CLIP)
# ----------------------------
def txt2img_sample(
    dit, rae,
    adapters: Dict[str, ControlAdapter],
    canny_img: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    control_weight: float,
    gamma_c: float,
    ctrl_gain: float,
    ctrl_inject: str,
    ctrl_block_scale: float,
    prompt: Optional[str],
    clip_guidance_scale: float,
    out_path: str,
):
    B = 1
    latent_shape = (B, 768, 32, 32)

    # start from pure noise
    z = torch.randn(latent_shape, device=device, dtype=dtype)
    t_start = 1.0

    # precompute ctrl tokens
    with torch.no_grad():
        s_ctrl_base = adapters["canny"](canny_img)  # [1,L,D]
        # 额外增益：让 canny 更“真控制”
        s_ctrl_base = s_ctrl_base * float(ctrl_gain)

    ts = torch.linspace(t_start, 0.0, steps + 1, device=device, dtype=dtype)

    # load clip if needed
    clip_pack = None
    if (prompt is not None) and (len(prompt.strip()) > 0) and (clip_guidance_scale > 0):
        clip_pack = _load_clip()

    for i in range(steps):
        t_curr = ts[i].repeat(B)
        t_next = ts[i+1].repeat(B)
        dt = (t_curr - t_next).view(B,1,1,1)  # >0

        # compute s_content with stronger control injection (per_block)
        # std-match so control magnitude follows content magnitude
        s_content_noctrl = build_s_content_enc_with_ctrl(
            dit, z, t_curr, y,
            s_ctrl=None,
            ctrl_inject="none",
            ctrl_block_scale=0.0,
        )

        sc = s_content_noctrl.std(dim=(1,2), keepdim=True)
        cc = s_ctrl_base.std(dim=(1,2), keepdim=True)
        s_ctrl = s_ctrl_base * (sc / (cc + 1e-6))

        s_content = build_s_content_enc_with_ctrl(
            dit, z, t_curr, y,
            s_ctrl=s_ctrl,
            ctrl_inject=ctrl_inject,
            ctrl_block_scale=ctrl_block_scale,
        )

        # control-cfg on/off
        s_on  = s_content + control_weight * s_ctrl
        v_on  = dit(z, t_curr, y, s=s_on, mask=None)
        v_off = dit(z, t_curr, y, s=s_content, mask=None)
        v = v_off + gamma_c * (v_on - v_off)

        # ODE update
        z = z - v * dt
        if torch.isnan(z).any():
            z = torch.nan_to_num(z)

        # optional CLIP guidance (semantic)
        if clip_pack is not None:
            z_ = z.detach().clone().requires_grad_(True)
            # decode to image
            img = rae.decode(z_).float().clamp(0, 1)
            loss = clip_guidance_grad(clip_pack, img, prompt)
            grad = torch.autograd.grad(loss, z_, retain_graph=False, create_graph=False)[0]
            # guidance step (small)
            z = (z_ - clip_guidance_scale * grad).detach()

    with torch.no_grad():
        img = rae.decode(z).float().clamp(0, 1)
        save_image(img, out_path)
        print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="infer", choices=["infer"])
    ap.add_argument("--ctrl_mode", type=str, default="canny", choices=["canny"])
    ap.add_argument("--canny", type=str, required=True)

    ap.add_argument("--class_id", type=int, default=207)
    ap.add_argument("--infer_steps", type=int, default=64)
    ap.add_argument("--infer_control_weight", type=float, default=0.6)
    ap.add_argument("--gamma_c", type=float, default=1.2)

    # make canny "real control"
    ap.add_argument("--ctrl_gain", type=float, default=4.0, help="Multiply control tokens to strengthen canny constraint.")
    ap.add_argument("--ctrl_inject", type=str, default="per_block", choices=["once","per_block"],
                    help="How to inject control into encoder: once or per_block.")
    ap.add_argument("--ctrl_block_scale", type=float, default=0.35,
                    help="Scale used when injecting s_ctrl into encoder blocks (in addition to infer_control_weight).")

    # semantic
    ap.add_argument("--prompt", type=str, default="", help="CLIP semantic prompt.")
    ap.add_argument("--clip_guidance_scale", type=float, default=0.0,
                    help=">0 enables CLIP guidance; try 1~8. Requires open_clip_torch.")

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.canny):
        raise SystemExit(f"Canny image not found: {args.canny}")

    out_train_dir = os.path.join(OUT_ROOT, "train", "adapter", "canny")
    out_infer_dir = ensure_dir(os.path.join(OUT_ROOT, "infer", "canny", "txt2img_semantic"))

    # load models
    manager = ModelManager(device=device)
    rae = manager.load_rae().eval()
    dit = manager.load_dit()
    if dit is None:
        raise RuntimeError("Failed to load DiT/DDT")
    dit = dit.to(device=device, dtype=dtype).eval()
    for p in rae.parameters(): p.requires_grad_(False)
    for p in dit.parameters(): p.requires_grad_(False)

    # load ckpt
    ckpt_path = os.path.join(out_train_dir, "latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = latest_ckpt(out_train_dir, pattern="ckpt_step_*.pt")
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise SystemExit(f"No ckpt found in {out_train_dir}. Train first.")
    adapters, ck = load_ckpt_and_build_adapters(dit, ckpt_path)

    # load canny ctrl image
    canny_img = _ensure_3ch(load_and_transform(args.canny, 448))

    y = torch.tensor([args.class_id], device=device, dtype=torch.long)

    ts = now_ts()
    tag = f"cid{args.class_id}_steps{args.infer_steps}_cw{args.infer_control_weight}_gc{args.gamma_c}_gain{args.ctrl_gain}"
    if args.clip_guidance_scale > 0 and len(args.prompt.strip()) > 0:
        tag += "_clip"
    out_path = os.path.join(out_infer_dir, f"gen_{tag}_{ts}.png")

    txt2img_sample(
        dit=dit, rae=rae,
        adapters=adapters,
        canny_img=canny_img,
        y=y,
        steps=args.infer_steps,
        control_weight=args.infer_control_weight,
        gamma_c=args.gamma_c,
        ctrl_gain=args.ctrl_gain,
        ctrl_inject=args.ctrl_inject,
        ctrl_block_scale=args.ctrl_block_scale,
        prompt=(args.prompt if len(args.prompt.strip()) > 0 else None),
        clip_guidance_scale=float(args.clip_guidance_scale),
        out_path=out_path,
    )

if __name__ == "__main__":
    main()
