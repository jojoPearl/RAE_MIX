import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import gc
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.ndimage as ndimage
import sys
import numpy as np
import datetime
import os
import json
from typing import Optional, Tuple, List, Dict
from rembg import remove

# -------------------- Path setup --------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from torchvision.utils import save_image
from src.semantic.modelManager import ModelManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_size = 448
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
# DTYPE = torch.float16 if device.type == "cuda" else torch.float32
DTYPE = torch.float32

_processor = None
_model = None

def _init_clipseg():
    global _processor, _model
    if _model is None or _processor is None:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        print(f"Initializing CLIPSeg model on {device}...")
        MODEL_ID = "CIDAS/clipseg-rd64-refined"
        _processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
        _model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to(device)
        _model.eval()

def extract_semantic_mask_with_clipseg(
    image: Image.Image, 
    target_text: str, 
    feature_size: Tuple[int, int], 
    threshold: float = 0.5
) -> torch.Tensor:
    _init_clipseg()
    inputs = _processor(
        text=[target_text],
        images=[image],
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = _model(**inputs)

    # 1. Get 0~1 probability map after Sigmoid
    preds = torch.sigmoid(outputs.logits).unsqueeze(1) 

    # 2. Interpolation scaling
    mask_tensor = F.interpolate(
        preds,
        size=feature_size,
        mode="bilinear",
        align_corners=False
    )

    # 3. Important improvement: if threshold <= 0, return raw probability map (for heatmap localization)
    # If threshold > 0, return binarized mask (for object extraction)
    if threshold <= 0:
        return mask_tensor.squeeze()
    
    return (mask_tensor.squeeze() > threshold).float()

def get_text_guided_coords(
    base_image,
    location_prompt: str,
    obj_h: int,
    obj_w: int,
    latent_hw: Tuple[int, int],
    avoid_mask: Optional[torch.Tensor] = None,   # [1,1,H,W] or [H,W] in [0,1]
    avoid_strength: float = 2.0,                 # kept for backward compatibility
    avoid_thr: float = 0.2,                      # occupied threshold
    avoid_dilate_iters: int = 4,                 # dilation for hard exclusion buffer
):
    """
    Returns (ts_h, te_h, ts_w, te_w) in latent coordinates or None.
    """
    H, W = latent_hw

    # 1) Raw probability map in latent resolution
    heatmap = extract_semantic_mask_with_clipseg(base_image, location_prompt, (H, W), 0.0)
    heatmap = heatmap.float().cpu().numpy()

    # 2) Validity check
    if heatmap.max() < 0.05:
        return None

    # 3) Smooth heatmap
    heatmap_smoothed = ndimage.gaussian_filter(heatmap, sigma=1.0)

    # 4) Apply avoidance mask (hard exclusion from occupied regions)
    if avoid_mask is not None:
        if isinstance(avoid_mask, torch.Tensor):
            am = avoid_mask.detach().float().cpu()
            if am.ndim == 4:
                am = am[0, 0]
            am = am.numpy()
        else:
            am = np.asarray(avoid_mask, dtype=np.float32)

        if am.shape != (H, W):
            # If caller passed mismatched size, best effort resize with numpy/scipy
            # (Usually caller will pass correct latent size.)
            am = ndimage.zoom(am, (H / am.shape[0], W / am.shape[1]), order=1)

        occ = (am > avoid_thr)
        if occ.max() > 0:
            if int(avoid_dilate_iters) > 0:
                occ = ndimage.binary_dilation(occ, iterations=int(avoid_dilate_iters))
            heatmap_smoothed = heatmap_smoothed.copy()
            heatmap_smoothed[occ] = 0.0
            if float(heatmap_smoothed.max()) < 1e-6:
                return None

    # 5) Find peak
    idx = int(np.argmax(heatmap_smoothed))
    y, x = idx // W, idx % W

    ts_h = int(np.clip(y - obj_h // 2, 0, H - obj_h))
    ts_w = int(np.clip(x - obj_w // 2, 0, W - obj_w))

    print(f"[Smart Placement] Found target center at: ({y}, {x})")
    return ts_h, ts_h + obj_h, ts_w, ts_w + obj_w


def load_and_transform(path: str, targemodel_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((targemodel_size, targemodel_size), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=DTYPE)
    return tensor

def cleanup_memory():
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
def _calculate_dynamic_coords(H: int, W: int, h_new: int, w_new: int, area: str) -> Tuple[int, int, int, int]:
    """
    Calculate placement coordinates on the background (H, W) based on the dynamic object size (h_new, w_new).
    """
    # Ensure object size does not exceed background size
    h_new = min(h_new, H)
    w_new = min(w_new, W)

    if area == 'top_left':
        ts_h, ts_w = 0, 0
    elif area == 'top_right':
        ts_h, ts_w = 0, W - w_new
    elif area == 'bottom_left':
        ts_h, ts_w = H - h_new, 0
    elif area == 'center':
        ts_h, ts_w = (H - h_new) // 2, (W - w_new) // 2
    elif area == 'bottom_right':
        ts_h, ts_w = H - h_new, W - w_new
    else:
        # Default to bottom-right
        ts_h, ts_w = H - h_new, W - w_new

    te_h, te_w = ts_h + h_new, ts_w + w_new
    
    # Final boundary check to prevent slice overflow
    te_h = min(te_h, H)
    te_w = min(te_w, W)
    
    return ts_h, te_h, ts_w, te_w

def place_with_avoidance(
    H: int,
    W: int,
    ph: int,
    pw: int,
    target_area: str,
    occupied_mask_latent: Optional[torch.Tensor],
    search_radius: int = 20,
    step: int = 4,
    occ_thr: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    Place patch by target_area, then locally search for minimal overlap with occupied mask.
    """
    ts_h0, te_h0, ts_w0, te_w0 = _calculate_dynamic_coords(H, W, ph, pw, target_area)
    ts_h0, te_h0, ts_w0, te_w0 = clamp_coords(ts_h0, te_h0, ts_w0, te_w0, H, W)

    if occupied_mask_latent is None:
        return ts_h0, te_h0, ts_w0, te_w0

    occ = occupied_mask_latent.detach()
    if occ.ndim == 4:
        occ = occ[0, 0]
    occ = (occ > occ_thr).to(dtype=torch.float32)

    search_radius = max(0, int(search_radius))
    step = max(1, int(step))

    best = (ts_h0, te_h0, ts_w0, te_w0)
    best_score = None

    for dh in range(-search_radius, search_radius + 1, step):
        for dw in range(-search_radius, search_radius + 1, step):
            ts_h = ts_h0 + dh
            ts_w = ts_w0 + dw
            te_h = ts_h + ph
            te_w = ts_w + pw
            ts_h, te_h, ts_w, te_w = clamp_coords(ts_h, te_h, ts_w, te_w, H, W)

            patch = occ[ts_h:te_h, ts_w:te_w]
            overlap = float(patch.mean().item()) if patch.numel() > 0 else 1.0

            # Prefer lower overlap while staying close to requested area.
            dist = float(abs(dh) + abs(dw)) / float(max(1, search_radius))
            score = overlap + 0.05 * dist

            if (best_score is None) or (score < best_score):
                best_score = score
                best = (ts_h, te_h, ts_w, te_w)

    return best

def apply_m1_scaling(
        img: Image.Image,
        scale_factor: float,
        targemodel_size: int = 224,
        background_mode: str = "mean"
) -> Image.Image:
    """
    M1 Method: Scaling in pixel space using PIL.
    Resizes the object and pads it into a fixed-size canvas.
    """
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h

    if aspect > 1:
        new_w = int(targemodel_size * scale_factor)
        new_h = int(new_w / aspect)
    else:
        new_h = int(targemodel_size * scale_factor)
        new_w = int(new_h * aspect)

    # Use LANCZOS for high-quality downsampling
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Handle background color calculation
    if background_mode == "mean":
        img_array = np.array(img_resized)
        bg_color = tuple(np.mean(img_array.reshape(-1, 3), axis=0).astype(int))
    elif background_mode == "edge":
        img_array = np.array(img_resized)
        edge_pixels = np.concatenate([
            img_array[0, :, :], img_array[-1, :, :], 
            img_array[:, 0, :], img_array[:, -1, :]
        ], axis=0)
        bg_color = tuple(np.mean(edge_pixels, axis=0).astype(int))
    else:
        bg_color = (0, 0, 0)

    # Create canvas and paste the resized image in the center
    new_img = Image.new("RGB", (targemodel_size, targemodel_size), bg_color)
    upper = (targemodel_size - new_h) // 2
    left = (targemodel_size - new_w) // 2
    new_img.paste(img_resized, (left, upper))

    return new_img

def expand_bbox(rmin, rmax, cmin, cmax, H, W, pad_ratio: float = 0.22, pad_px: int = 18) -> Tuple[int, int, int, int]:
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    pad = int(max(pad_px, pad_ratio * max(h, w)))
    r0 = max(0, rmin - pad)
    r1 = min(H - 1, rmax + pad)
    c0 = max(0, cmin - pad)
    c1 = min(W - 1, cmax + pad)
    return r0, r1, c0, c1

def feather_mask(mask: torch.Tensor, iters: int = 1, k: int = 3) -> torch.Tensor:
    """
    Simple feather (blur) using avg_pool2d.
    mask: [B,1,H,W] float in [0,1]
    """
    m = mask
    pad = k // 2
    for _ in range(iters):
        m = F.avg_pool2d(m, kernel_size=k, stride=1, padding=pad)
    # Keep more edge detail and avoid over-soft transparency.
    m = 0.7 * m + 0.3 * mask
    return m.clamp(0, 1)


def bbox_from_mask(mask: torch.Tensor, thr: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
    """
    mask: [1,1,H,W] in [0,1]
    return (y0,y1,x0,x1) inclusive
    """
    m = (mask[0, 0] > thr).detach().cpu().numpy()
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return y0, y1, x0, x1


def clamp_coords(ts_h, te_h, ts_w, te_w, H, W) -> Tuple[int, int, int, int]:
    """
    Clamp coords to image bounds.
    """
    ts_h = max(0, min(ts_h, H))
    te_h = max(0, min(te_h, H))
    ts_w = max(0, min(ts_w, W))
    te_w = max(0, min(te_w, W))
    # ensure non-empty
    if te_h <= ts_h:
        te_h = min(H, ts_h + 1)
    if te_w <= ts_w:
        te_w = min(W, ts_w + 1)
    return ts_h, te_h, ts_w, te_w


def safe_patch_slice(t: torch.Tensor, ts_h, te_h, ts_w, te_w) -> torch.Tensor:
    return t[:, :, ts_h:te_h, ts_w:te_w]


# -------------------------------------------------------------------------
# Core: latent canvas scaling using grid_sample (STN-style)
# -------------------------------------------------------------------------
def scale_in_latent_canvas(
    feat: torch.Tensor,
    mask: torch.Tensor,
    scale_factor: float,
    padding_mode: str = "zeros"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center scaling on a fixed-size latent canvas.
    feat: [B,C,H,W], mask: [B,1,H,W]
    scale_factor > 1 => object bigger
    """
    B, C, H, W = feat.shape
    a = 1.0 / max(scale_factor, 1e-6)  # output->input map

    theta = torch.tensor(
        [[[a, 0.0, 0.0],
          [0.0, a, 0.0]]],
        device=feat.device,
        dtype=feat.dtype
    ).repeat(B, 1, 1)

    grid = F.affine_grid(theta, size=feat.size(), align_corners=False)
    feat_s = F.grid_sample(feat, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)

    grid_m = F.affine_grid(theta, size=mask.size(), align_corners=False)
    mask_s = F.grid_sample(mask, grid_m, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    mask_s = mask_s.clamp(0, 1)
    return feat_s, mask_s


# -------------------------------------------------------------------------
# 1) Get cropped object + soft mask in pixel space (M1 fixed = 1.0 for encode)
# -------------------------------------------------------------------------
def get_cropped_object_tensor(
    raw_image: Image.Image,
    target_text: str,
    scale_factor: float = 1.0,   # keep 1.0 for quality encode
    background_mode: str = "mean",
    target_size_for_encoder: int = model_size,
    clipseg_threshold: float = 0.4,
):
    """
    Returns:
      obj_tensor: [1,3,model_size,model_size]
      mask_tensor: [1,1,model_size,model_size]
    """
    rgba = remove(raw_image.convert("RGBA"))
    rgba_np = np.array(rgba).astype(np.uint8)

    rgb = rgba_np[:, :, :3]
    a = rgba_np[:, :, 3].astype(np.float32) / 255.0

    thr = 0.005
    m = (a > thr).astype(np.uint8)
    m = ndimage.binary_dilation(m, iterations=3)
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m).astype(np.uint8)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        obj_to_scale = raw_image.convert("RGB")
        mask_to_scale = Image.fromarray(
            (np.ones((raw_image.size[1], raw_image.size[0])) * 255).astype(np.uint8),
            mode="L",
        )
    else:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        H, W = m.shape
        h = y1 - y0 + 1
        w = x1 - x0 + 1
        pad = int(max(24, 0.30 * max(h, w)))
        y0 = max(0, y0 - pad)
        y1 = min(H - 1, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(W - 1, x1 + pad)

        obj_crop = rgb[y0:y1 + 1, x0:x1 + 1]
        mask_crop = (m[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)

        obj_to_scale = Image.fromarray(obj_crop, mode="RGB")
        mask_to_scale = Image.fromarray(mask_crop, mode="L")

    final_obj_pil = apply_m1_scaling(
        obj_to_scale.convert("RGB"),
        scale_factor,
        target_size_for_encoder,
        background_mode,
    )
    final_mask_pil = apply_m1_scaling(
        mask_to_scale.convert("RGB"),
        scale_factor,
        target_size_for_encoder,
        background_mode="black",
    ).convert("L")

    obj_tensor = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device=device, dtype=DTYPE)
    mask_tensor = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device=device, dtype=DTYPE)
    if mask_tensor.shape[1] > 1:
        mask_tensor = mask_tensor[:, 0:1]

    mask_tensor = mask_tensor.clamp(0, 1)
    return obj_tensor, mask_tensor


# -------------------------------------------------------------------------
# 2) Semantic fusion (NEW): scale in latent canvas -> bbox from scaled mask -> patch blend
# -------------------------------------------------------------------------
@torch.no_grad()
def semantic_fusion_v2(
    canvas_latent: torch.Tensor,          # [B,C,H,W] current canvas
    object_latent: torch.Tensor,          # [B,C,H,W] object canvas latent (same shape)
    object_mask_img: torch.Tensor,        # [B,1,model_size,model_size] mask in image space
    raw_base_image: Image.Image,
    location_prompt: Optional[str],
    target_area: str,
    scale_factor: float,
    use_smart_placement: bool = True,
    mask_thr_bbox: float = 0.10,
    feather_iters: int = 1,
    feather_k: int = 3,
    overlap_mode: str = "no_overwrite",   # "allow" | "no_overwrite" | "alpha"
    occupied_mask_latent: Optional[torch.Tensor] = None,  # [B,1,H,W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      new_canvas_latent
      object_union_mask_latent (mask at paste location in latent resolution)
    """
    B, C, H, W = canvas_latent.shape

    # mask to latent res
    mask_latent = F.interpolate(object_mask_img, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)

    # 1) Extract bbox patch at original latent scale first.
    bb0 = bbox_from_mask(mask_latent, thr=float(mask_thr_bbox))
    if bb0 is None:
        return canvas_latent, torch.zeros((B, 1, H, W), device=canvas_latent.device, dtype=canvas_latent.dtype)

    y0, y1, x0, x1 = bb0
    obj_patch = object_latent[:, :, y0:y1 + 1, x0:x1 + 1]       # [B,C,ph,pw]
    mask_patch = mask_latent[:, :, y0:y1 + 1, x0:x1 + 1]        # [B,1,ph,pw]

    # 2) Scale only the bbox patch (less interpolation pollution than full-canvas scaling).
    ph, pw = obj_patch.shape[-2:]
    new_ph = max(1, int(ph * scale_factor))
    new_pw = max(1, int(pw * scale_factor))
    obj_patch = F.interpolate(obj_patch, size=(new_ph, new_pw), mode="bicubic", align_corners=False, antialias=True)
    mask_scaled = F.interpolate(mask_patch, size=(new_ph, new_pw), mode="bilinear", align_corners=False).clamp(0, 1)

    feather_k = max(1, int(feather_k))
    if feather_k % 2 == 0:
        feather_k += 1

    # 3) core-hard + thin-soft boundary
    core = (mask_scaled > 0.35).to(mask_scaled.dtype)
    soft = feather_mask(mask_scaled, iters=1, k=3)
    soft = (soft ** 1.6).clamp(0, 1)
    m_patch = torch.maximum(core, soft)
    core_patch = core

    # Tighten scaled patch with hard core bbox (avoid bloated soft extent).
    bb_tight = bbox_from_mask(core_patch, thr=0.5)
    if bb_tight is None:
        return canvas_latent, torch.zeros((B, 1, H, W), device=canvas_latent.device, dtype=canvas_latent.dtype)
    yy0, yy1, xx0, xx1 = bb_tight
    obj_patch = obj_patch[:, :, yy0:yy1 + 1, xx0:xx1 + 1]
    m_patch = m_patch[:, :, yy0:yy1 + 1, xx0:xx1 + 1]
    core_patch = core_patch[:, :, yy0:yy1 + 1, xx0:xx1 + 1]
    ph, pw = obj_patch.shape[-2:]

    # 4) choose paste location (forced area-based placement + avoidance search)
    ts_h, te_h, ts_w, te_w = place_with_avoidance(
        H=H,
        W=W,
        ph=ph,
        pw=pw,
        target_area=target_area,
        occupied_mask_latent=occupied_mask_latent,
        search_radius=24,
        step=4,
        occ_thr=0.2,
    )

    # Legacy smart-placement logic kept for reference (disabled by request).
    # effective_smart = False
    # if use_smart_placement and location_prompt and raw_base_image:
    #     coords = get_text_guided_coords(
    #         raw_base_image,
    #         location_prompt,
    #         ph,
    #         pw,
    #         (H, W),
    #         avoid_mask=occupied_mask_latent,
    #         avoid_strength=2.0,
    #         avoid_thr=0.2,
    #     )
    #     if coords:
    #         ts_h, te_h, ts_w, te_w = coords
    #         effective_smart = True
    #
    # if not effective_smart:
    #     ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, ph, pw, target_area)

    # clamp coords
    ts_h, te_h, ts_w, te_w = clamp_coords(ts_h, te_h, ts_w, te_w, H, W)

    # get base patch
    base_patch = safe_patch_slice(canvas_latent, ts_h, te_h, ts_w, te_w)

    # 5) size align
    if base_patch.shape[-2:] != obj_patch.shape[-2:]:
        # Single high-quality resize right before paste to reduce blur accumulation.
        obj_patch = F.interpolate(
            obj_patch,
            size=base_patch.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        m_patch = F.interpolate(
            m_patch,
            size=base_patch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        core_patch = F.interpolate(
            core_patch,
            size=base_patch.shape[-2:],
            mode="nearest",
        )

    # 6) overlap control
    # occupied_mask_latent: regions already occupied by previous objects
    if overlap_mode not in ("allow", "no_overwrite", "alpha"):
        overlap_mode = "no_overwrite"

    if occupied_mask_latent is None:
        occ_patch = None
    else:
        occ_patch = safe_patch_slice(occupied_mask_latent, ts_h, te_h, ts_w, te_w).clamp(0, 1)

    # effective mask for blending
    m_eff = m_patch
    core_eff = core_patch

    if overlap_mode == "no_overwrite" and occ_patch is not None:
        # do not write where already occupied
        m_eff = m_eff * (1.0 - (occ_patch > 0.2).to(dtype=m_eff.dtype))
        core_eff = core_eff * (1.0 - (occ_patch > 0.2).to(dtype=core_eff.dtype))

    elif overlap_mode == "alpha" and occ_patch is not None:
        # if overlap, reduce new object's alpha a bit
        m_eff = m_eff * (1.0 - 0.5 * occ_patch.to(dtype=m_eff.dtype))
        core_eff = core_eff * (1.0 - 0.5 * occ_patch.to(dtype=core_eff.dtype))

    # 7) residual blend + hard overwrite in core
    blend_strength = 1.0
    blended_patch = base_patch + m_eff * blend_strength * (obj_patch - base_patch)
    hard_core = (core_eff > 0.8).expand(-1, C, -1, -1)
    blended_patch = torch.where(hard_core, obj_patch, blended_patch)

    new_canvas = canvas_latent.clone()
    new_canvas[:, :, ts_h:te_h, ts_w:te_w] = blended_patch

    # 8) produce union mask at latent res in final paste position
    union_mask = torch.zeros((B, 1, H, W), device=new_canvas.device, dtype=new_canvas.dtype)
    union_mask[:, :, ts_h:te_h, ts_w:te_w] = m_eff.clamp(0, 1)

    return new_canvas, union_mask


# -------------------------------------------------------------------------
# 3) Multi-Object Composition (Stage1)
# -------------------------------------------------------------------------
def stage1_composition(
    base_image_path: str,
    objects_list: List[Dict],
    output_path: str,
    fused_path_prefix: str,
    global_scale_factors: List[float] = [1.0],
    overlap_mode: str = "no_overwrite",   # "allow" | "no_overwrite" | "alpha"
):
    os.makedirs(output_path, exist_ok=True)

    # Load base
    try:
        base_img_tensor = load_and_transform(base_image_path, model_size).to(device=device)
        base_pil_image = Image.open(base_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Base image file not found: {base_image_path}")
        return

    # Sort by z_order (small -> large, or vice versa; choose your convention)
    # Here: smaller z_order fused first, larger fused later (later may cover earlier)
    objects_sorted = sorted(objects_list, key=lambda d: d.get("z_order", 0))

    # Load model once
    manager = ModelManager(device=device)
    rae = manager.load_rae()
    cleanup_memory()

    print(f"--- [Stage1] Encoding base once ---")
    with torch.amp.autocast('cuda', dtype=DTYPE):
        base_latent_init = rae.encode(base_img_tensor)   # [1,C,H,W]

    # Pre-encode objects once
    encoded_objects = []
    print(f"--- [Stage1] Pre-encoding {len(objects_sorted)} objects ---")
    for conf in objects_sorted:
        try:
            r_pil = Image.open(conf["path"]).convert("RGB")

            # M1 fixed=1.0 for encode quality
            obj_tensor, obj_mask = get_cropped_object_tensor(
                raw_image=r_pil,
                target_text=conf["text"],
                scale_factor=1.0,
                background_mode="mean",
                clipseg_threshold=float(conf.get("clipseg_threshold", 0.4)),
            )

            with torch.amp.autocast('cuda', dtype=DTYPE):
                obj_latent = rae.encode(obj_tensor)

            encoded_objects.append({
                "latent": obj_latent,
                "mask_img": obj_mask,
                "config": conf
            })
        except FileNotFoundError:
            print(f"[Warning] Object image not found: {conf.get('path')}")
            continue

    # Compose for each global scale
    for gscale in global_scale_factors:
        print(f"\n--- [Stage1] Global scale = {gscale} ---")
        canvas_latent = base_latent_init.clone()

        # track occupied mask in latent res (for overlap control)
        B, C, H, W = canvas_latent.shape
        occupied = torch.zeros((B, 1, H, W), device=device, dtype=canvas_latent.dtype)
        union_mask_total = torch.zeros((B, 1, H, W), device=device, dtype=canvas_latent.dtype)

        with torch.amp.autocast('cuda', dtype=DTYPE):
            for item in encoded_objects:
                conf = item["config"]
                obj_lat = item["latent"]
                obj_mask_img = item["mask_img"]

                base_scale = conf.get("base_scale", 1.0)
                final_scale = float(gscale * base_scale)

                print(f"  + Fuse '{conf['text']}' scale={final_scale:.2f} area={conf.get('target_area','center')} z={conf.get('z_order',0)}")

                canvas_latent, union_mask = semantic_fusion_v2(
                    canvas_latent=canvas_latent,
                    object_latent=obj_lat,
                    object_mask_img=obj_mask_img,
                    raw_base_image=base_pil_image,
                    location_prompt=conf.get("location_prompt"),
                    target_area=conf.get("target_area", "center"),
                    scale_factor=final_scale,
                    use_smart_placement=conf.get("use_smart", True),
                    mask_thr_bbox=float(conf.get("mask_thr_bbox", 0.10)),
                    feather_iters=int(conf.get("feather_iters", 1)),
                    feather_k=int(conf.get("feather_k", 3)),
                    overlap_mode=overlap_mode,
                    occupied_mask_latent=occupied
                )

                # update masks
                union_mask_total = (union_mask_total + union_mask).clamp(0, 1)
                occupied = torch.maximum(occupied, (union_mask > 0.25).to(dtype=occupied.dtype))

        # Save pt + preview
        dir_name = os.path.dirname(fused_path_prefix)
        base_name = os.path.basename(fused_path_prefix).replace(".pt", "")
        final_pt_name = f"{base_name}_composition_scale_{gscale}.pt"
        final_pt_path = os.path.join(dir_name, final_pt_name)
        os.makedirs(dir_name, exist_ok=True)
        union_mask_core = (union_mask_total > 0.35).to(dtype=union_mask_total.dtype)

        torch.save({
            "fused_features": canvas_latent.detach().cpu(),
            "union_mask_latent": union_mask_total.detach().cpu(),
            "union_mask_core_latent": union_mask_core.detach().cpu(),
            "occupied_mask_latent": occupied.detach().cpu(),
            "global_scale": gscale,
            "objects_info": objects_sorted,
            "base_image": base_image_path,
            "timestamp": timestamp,
            "overlap_mode": overlap_mode
        }, final_pt_path)

        # decode preview
        with torch.no_grad():
            check_img = rae.decode(canvas_latent)
        check_path = os.path.join(output_path, f"check_composition_scale_{gscale}_{timestamp}.png")
        save_image(check_img.float(), check_path)

        # also save union mask as image for debug
        mask_vis = union_mask_total[0].repeat(3, 1, 1).detach().cpu()  # [3,H,W]
        save_image(mask_vis, os.path.join(output_path, f"mask_union_scale_{gscale}_{timestamp}.png"))

        print(f"  Saved: {final_pt_path}")
        print(f"  Preview: {check_path}")

    del rae
    cleanup_memory()
    print("\n--- [Stage1] Composition Completed ---")


# -------------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    group_dir = os.path.join(project_root, "assets", "group0")
    base_image_path = os.path.join(group_dir, "image.png")
    output_path = os.path.join(group_dir, "stage1_result1")
    fused_path = os.path.join(output_path, "fused_results.pt")
    os.makedirs(output_path, exist_ok=True)

    objects_json_path = os.path.join(group_dir, "objects.json")
    with open(objects_json_path, "r", encoding="utf-8") as f:
        objects_to_add = json.load(f)
    for item in objects_to_add:
        p = item.get("path")
        if p and not os.path.isabs(p):
            item["path"] = os.path.join(project_root, p)

    # global_scale_factors = [0.3, 0.35, 0.4, 0.45, 0.5]
    global_scale_factors = [0.8,0.9,0.6]

    stage1_composition(
        base_image_path=base_image_path,
        objects_list=objects_to_add,
        output_path=output_path,
        fused_path_prefix=fused_path,
        global_scale_factors=global_scale_factors,
        overlap_mode="no_overwrite"
    )
