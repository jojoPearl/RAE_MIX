import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import sys
import numpy as np
import datetime
import os
from typing import Optional, Tuple, List, Dict

# -------------------- Path setup --------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# -------------------- Imports --------------------
from torchvision.utils import save_image
from src.semantic.segmentation import get_text_guided_coords, extract_semantic_mask_with_clipseg
from src.semantic.utils import cleanup_memory, load_and_transform, _calculate_dynamic_coords
from src.semantic.resize import apply_m1_scaling
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE_MODERN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
t_size = 448
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

# -------------------------------------------------------------------------
# Utils: mask feather + bbox + clipping
# -------------------------------------------------------------------------
def feather_mask(mask: torch.Tensor, iters: int = 2, k: int = 3) -> torch.Tensor:
    """
    Simple feather (blur) using avg_pool2d.
    mask: [B,1,H,W] float in [0,1]
    """
    m = mask
    pad = k // 2
    for _ in range(iters):
        m = F.avg_pool2d(m, kernel_size=k, stride=1, padding=pad)
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
    target_size_for_encoder: int = 448,
    clipseg_threshold: float = 0.3,
):
    """
    Returns:
      obj_tensor: [1,3,t_size,t_size]
      mask_tensor: [1,1,t_size,t_size] soft in [0,1]
    """
    W_orig, H_orig = raw_image.size

    # CLIPSeg soft mask (H_orig, W_orig)
    mask_soft = extract_semantic_mask_with_clipseg(
        image=raw_image,
        target_text=target_text,
        feature_size=(H_orig, W_orig),
        threshold=clipseg_threshold
    ).cpu().squeeze().clamp(0, 1).numpy()

    # bbox using threshold (only for cropping)
    binary = (mask_soft > clipseg_threshold).astype(np.uint8)
    rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)

    if not np.any(rows) or not np.any(cols):
        print(f"[Warning] No object '{target_text}' found. Using full image.")
        obj_to_scale = raw_image
        mask_to_scale = Image.fromarray((np.ones((H_orig, W_orig)) * 255).astype(np.uint8), mode="L")
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        obj_to_scale = Image.fromarray(np.array(raw_image)[rmin:rmax + 1, cmin:cmax + 1])
        mask_crop = (mask_soft[rmin:rmax + 1, cmin:cmax + 1] * 255.0).astype(np.uint8)
        mask_to_scale = Image.fromarray(mask_crop, mode="L")

    # M1 scaling for encoder input (usually keep scale_factor=1.0)
    final_obj_pil = apply_m1_scaling(obj_to_scale.convert("RGB"), scale_factor, target_size_for_encoder, background_mode)
    final_mask_pil = apply_m1_scaling(mask_to_scale.convert("RGB"), scale_factor, target_size_for_encoder, background_mode="black").convert("L")

    obj_tensor = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device=device, dtype=DTYPE_MODERN)
    mask_tensor = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device=device, dtype=DTYPE_MODERN)
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
    object_mask_img: torch.Tensor,        # [B,1,t_size,t_size] mask in image space
    raw_base_image: Image.Image,
    location_prompt: Optional[str],
    target_area: str,
    scale_factor: float,
    use_smart_placement: bool = True,
    mask_thr_bbox: float = 0.2,
    feather_iters: int = 2,
    feather_k: int = 3,
    overlap_mode: str = "allow",          # "allow" | "no_overwrite" | "alpha"
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

    # 1) scale object+mask on latent canvas
    obj_scaled, mask_scaled = scale_in_latent_canvas(object_latent, mask_latent, scale_factor=scale_factor)

    # 2) feather mask
    mask_scaled = feather_mask(mask_scaled, iters=feather_iters, k=feather_k)

    # 3) bbox from scaled mask (this bbox changes with scale)
    bb = bbox_from_mask(mask_scaled, thr=mask_thr_bbox)
    if bb is None:
        return canvas_latent, torch.zeros((B, 1, H, W), device=canvas_latent.device, dtype=canvas_latent.dtype)

    y0, y1, x0, x1 = bb
    obj_patch = obj_scaled[:, :, y0:y1 + 1, x0:x1 + 1]          # [B,C,ph,pw]
    m_patch   = mask_scaled[:, :, y0:y1 + 1, x0:x1 + 1]         # [B,1,ph,pw]
    ph, pw = obj_patch.shape[-2:]

    # 4) choose paste location
    effective_smart = False
    if use_smart_placement and location_prompt and raw_base_image:
        coords = get_text_guided_coords(raw_base_image, location_prompt, ph, pw, (H, W))
        if coords:
            ts_h, te_h, ts_w, te_w = coords
            effective_smart = True

    if not effective_smart:
        ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, ph, pw, target_area)

    # clamp coords
    ts_h, te_h, ts_w, te_w = clamp_coords(ts_h, te_h, ts_w, te_w, H, W)

    # get base patch
    base_patch = safe_patch_slice(canvas_latent, ts_h, te_h, ts_w, te_w)

    # 5) size align
    if base_patch.shape[-2:] != obj_patch.shape[-2:]:
        obj_patch = F.interpolate(obj_patch, size=base_patch.shape[-2:], mode="bilinear", align_corners=False)
        m_patch   = F.interpolate(m_patch,   size=base_patch.shape[-2:], mode="bilinear", align_corners=False)

    # 6) overlap control
    # occupied_mask_latent: regions already occupied by previous objects
    if overlap_mode not in ("allow", "no_overwrite", "alpha"):
        overlap_mode = "allow"

    if occupied_mask_latent is None:
        occ_patch = None
    else:
        occ_patch = safe_patch_slice(occupied_mask_latent, ts_h, te_h, ts_w, te_w).clamp(0, 1)

    # effective mask for blending
    m_eff = m_patch

    if overlap_mode == "no_overwrite" and occ_patch is not None:
        # do not write where already occupied
        m_eff = m_eff * (1.0 - (occ_patch > 0.2).float())

    elif overlap_mode == "alpha" and occ_patch is not None:
        # if overlap, reduce new object's alpha a bit
        m_eff = m_eff * (1.0 - 0.5 * occ_patch)

    # 7) blend
    new_canvas = canvas_latent.clone()
    new_canvas[:, :, ts_h:te_h, ts_w:te_w] = m_eff * obj_patch + (1.0 - m_eff) * base_patch

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
    overlap_mode: str = "allow",   # "allow" | "no_overwrite" | "alpha"
):
    os.makedirs(output_path, exist_ok=True)

    # Load base
    try:
        base_img_tensor = load_and_transform(base_image_path, t_size).to(device=device)
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
    with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
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
                target_size_for_encoder=t_size
            )

            with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
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

        with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
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
                    overlap_mode=overlap_mode,
                    occupied_mask_latent=occupied
                )

                # update masks
                union_mask_total = (union_mask_total + union_mask).clamp(0, 1)
                occupied = (occupied + union_mask).clamp(0, 1)

        # Save pt + preview
        dir_name = os.path.dirname(fused_path_prefix)
        base_name = os.path.basename(fused_path_prefix).replace(".pt", "")
        final_pt_name = f"{base_name}_composition_scale_{gscale}.pt"
        final_pt_path = os.path.join(dir_name, final_pt_name)
        os.makedirs(dir_name, exist_ok=True)

        torch.save({
            "fused_features": canvas_latent.detach().cpu(),
            "union_mask_latent": union_mask_total.detach().cpu(),
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
    base_image_path = "assets/group1/base.png"
    output_path = "assets/group1/stage1_result1/"
    fused_path = os.path.join(output_path, "fused_results.pt")
    os.makedirs(output_path, exist_ok=True)

    objects_to_add = [
        {
            "path": "assets/group1/r.png",
            "text": "dog",
            "target_area": "bottom_left",
            "location_prompt": None,
            "base_scale": 0.7,
            "use_smart": False,
            "z_order": 0
        },
        {
            "path": "assets/group1/r2.png",
            "text": "dog",
            "target_area": "bottom_right",
            "location_prompt": "central",
            "base_scale": 0.7,        
            "use_smart": False,      
            "z_order": 1
        }
    ]

    base_image_path = "assets/group4/base.png"
    output_path = "assets/group4/stage1_result1/"
    fused_path = os.path.join(output_path, "fused_results.pt")
    os.makedirs(output_path, exist_ok=True)

    objects_to_add = [
        {
            "path": "assets/group4/r.png",
            "text": "butterfly",
            "target_area": "bottom_left",
            "location_prompt": "on the flower",
            "base_scale": 0.8,
            "use_smart": True,
            "z_order": 0
        },
        {
            "path": "assets/group4/r2.png",
            "text": "butterfly",
            "target_area": "bottom_right",
            "location_prompt": "central",
            "base_scale": 0.8,        
            "use_smart": False,      
            "z_order": 1
        }
    ]


    global_scale_factors = [0.3, 0.35, 0.4, 0.45, 0.5]

    stage1_composition(
        base_image_path=base_image_path,
        objects_list=objects_to_add,
        output_path=output_path,
        fused_path_prefix=fused_path,
        global_scale_factors=global_scale_factors,
        overlap_mode="allow"   # try: "no_overwrite" or "alpha"
    )
