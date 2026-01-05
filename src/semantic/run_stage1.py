import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda"
)
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import sys
from typing import Tuple
import numpy as np
import datetime
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.stage1.rae import RAE
from torchvision.utils import save_image
from src.semantic.segmentation import get_text_guided_coords, extract_semantic_mask_with_clipseg
from src.semantic.utils import cleanup_memory, load_and_transform, _calculate_dynamic_coords
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE, DTYPE_MODERN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

t_size = 448

# --- [NEW TOOL]: Color Matching (From your proposed solution) ---
def match_mean_std(src, ref):
    """
    Adjusts the statistics of the src tensor to match the ref tensor.
    Helps the object blend into the lighting of the background.
    """
    # src, ref: [B, C, H, W]
    src_mean = src.mean([2, 3], keepdim=True)
    src_std  = src.std([2, 3], keepdim=True)
    ref_mean = ref.mean([2, 3], keepdim=True)
    ref_std  = ref.std([2, 3], keepdim=True)
    return (src - src_mean) / (src_std + 1e-6) * ref_std + ref_mean

def get_cropped_object_tensor(
        raw_image: Image.Image,
        target_text: str,
        scale_factor: float = 1.0,
        background_mode: str = "black", 
        target_size_for_encoder: int = 448
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    W_orig, H_orig = raw_image.size

    # 1. Mask Generation
    mask = extract_semantic_mask_with_clipseg(
        image=raw_image,
        target_text=target_text,
        feature_size=(H_orig, W_orig),
        threshold=0.3
    ).cpu().squeeze()

    # [STRATEGY]: Keep Hard Mask in Pixel Space.
    # Do NOT blur here, or the green background will bleed into the rabbit.
    binary_mask_np = (mask.numpy() > 0.3).astype(np.uint8)

    rows, cols = np.any(binary_mask_np, axis=1), np.any(binary_mask_np, axis=0)

    # 2. Crop
    if not np.any(rows) or not np.any(cols):
        print(f"[Warning] No object '{target_text}' found. Using full image.")
        obj_crop = raw_image
        mask_crop = Image.new("L", (W_orig, H_orig), 255)
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        buffer = 5
        rmin = max(0, rmin - buffer)
        rmax = min(H_orig, rmax + buffer)
        cmin = max(0, cmin - buffer)
        cmax = min(W_orig, cmax + buffer)

        obj_crop = Image.fromarray(np.array(raw_image)[rmin:rmax, cmin:cmax])
        mask_crop = Image.fromarray((binary_mask_np[rmin:rmax, cmin:cmax] * 255).astype(np.uint8), mode='L')

    # 3. Scale
    w_crop, h_crop = obj_crop.size
    max_dim = max(w_crop, h_crop)
    target_inner_size = int(target_size_for_encoder * scale_factor)
    resize_ratio = target_inner_size / max_dim
    new_w = int(w_crop * resize_ratio)
    new_h = int(h_crop * resize_ratio)

    obj_resized = obj_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Use NEAREST to maintain the "Clean Cut"
    mask_resized = mask_crop.resize((new_w, new_h), Image.Resampling.NEAREST)

    # 4. Paste on Black
    final_obj_pil = Image.new("RGB", (target_size_for_encoder, target_size_for_encoder), (0, 0, 0))
    final_mask_pil = Image.new("L", (target_size_for_encoder, target_size_for_encoder), 0)

    paste_x = (target_size_for_encoder - new_w) // 2
    paste_y = (target_size_for_encoder - new_h) // 2

    final_obj_pil.paste(obj_resized, (paste_x, paste_y), mask_resized)
    final_mask_pil.paste(mask_resized, (paste_x, paste_y))

    # 5. Tensor
    obj_tensor = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device, dtype=dtype)
    
    # Binarize strict
    mask_tensor = (mask_tensor > 0.5).float()

    if mask_tensor.shape[1] > 1:
        mask_tensor = mask_tensor[:, 0:1, :, :]

    return obj_tensor, mask_tensor


def semantic_fusion(
    base_latent: torch.Tensor,
    object_latent: torch.Tensor,
    object_mask: torch.Tensor,
    raw_base_image: Image.Image = None,
    location_prompt: str = None,
    target_area: str = "bottom_left",
    min_size: int = 8,
    scale_factor: float = 1.0,
    use_smart_placement: bool = True
) -> torch.Tensor:
    B, C, H, W = base_latent.shape
    
    # [OPTIONAL]: Adjusted base_unit size from your suggestion (0.35)
    # If the user feels the rabbit is too big, this helps.
    if location_prompt and "foreground" in location_prompt:
        base_unit = min(H, W) * 0.4 
    else:
        base_unit = min(H, W) * 0.6

    target_size = max(min_size, int(base_unit * scale_factor))
    target_size = min(target_size, min(H, W) - 2)

    # Use Bilinear for Latent Mask (This provides the softness safely)
    mask_latent = F.interpolate(object_mask, size=(H, W), mode="bilinear", align_corners=False)
    
    if mask_latent.max() < 0.1: return base_latent

    mask_np = (mask_latent[0, 0] > 0.5).cpu().numpy()
    ys, xs = np.where(mask_np)
    if len(xs) == 0: return base_latent

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    obj_crop = object_latent[:, :, y0:y1 + 1, x0:x1 + 1]
    mask_crop = mask_latent[:, :, y0:y1 + 1, x0:x1 + 1]

    h0, w0 = obj_crop.shape[2:]
    scale = target_size / max(h0, w0)
    h1, w1 = max(1, int(h0 * scale)), max(1, int(w0 * scale))

    obj_rs = F.interpolate(obj_crop, (h1, w1), mode="bicubic")
    mask_rs = F.interpolate(mask_crop, (h1, w1), mode="bilinear", align_corners=False)

    effective_smart = False
    if use_smart_placement and location_prompt and raw_base_image:
        coords = get_text_guided_coords(raw_base_image, location_prompt, h1, w1, (H, W))
        if coords:
            ts_h, te_h, ts_w, te_w = coords
            effective_smart = True
    
    if not effective_smart:
        ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, h1, w1, target_area)

    fused = base_latent.clone()
    base_patch = base_latent[:, :, ts_h:te_h, ts_w:te_w]
    
    if obj_rs.shape[2:] != base_patch.shape[2:]:
        obj_rs = F.interpolate(obj_rs, base_patch.shape[2:], mode="bicubic")
        mask_rs = F.interpolate(mask_rs, base_patch.shape[2:], mode="bilinear")

    # [KEY LOGIC]: Contrast Stretching
    # Safe Softness. Removes low-confidence background, keeps high-confidence rabbit solid.
    # Gamma 1.5 helps darken the semi-transparent edges (anti-halo).
    mask_rs = torch.clamp((mask_rs - 0.2) * 2.0, 0, 1) ** 1.5

    fused[:, :, ts_h:te_h, ts_w:te_w] = (
        mask_rs * obj_rs + (1.0 - mask_rs) * base_patch
    )
    return fused


@torch.inference_mode()
def stage1_extract_features(rae, base_img_tensor, base_pil_image, replace_pil_image, 
                           scale_factor, target_area, target_text, output_path,
                           location_prompt=None, use_smart_placement=True):
    
    object_tensor, object_mask = get_cropped_object_tensor(
        raw_image=replace_pil_image,
        target_text=target_text,
        scale_factor=scale_factor,
        target_size_for_encoder=t_size,
        background_mode="black" 
    )
    
    object_tensor = object_tensor.to(device, dtype=DTYPE_MODERN)
    object_mask = object_mask.to(device, dtype=DTYPE_MODERN)

    # [INTEGRATION]: Apply Color Matching Here!
    # This aligns the rabbit's color stats with the background scene.
    object_tensor = match_mean_std(object_tensor, base_img_tensor)

    with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
        base_latent = rae.encode(base_img_tensor)
        object_latent = rae.encode(object_tensor)

        fused_latent_2d = semantic_fusion(
            base_latent,
            object_latent,
            object_mask,
            raw_base_image=base_pil_image,
            location_prompt=location_prompt,
            target_area=target_area,
            scale_factor=1.0,
            min_size=8,
            use_smart_placement=use_smart_placement
        )
        
        fused_features_seq = fused_latent_2d.flatten(2).transpose(1, 2).contiguous()
        
        check_img = rae.decode(fused_latent_2d)
        os.makedirs(os.path.join(output_path), exist_ok=True)
        save_path = os.path.join(output_path, f"fusion_check_{timestamp}.png")
        save_image(check_img.float(), save_path)

    return fused_features_seq, fused_latent_2d


def stage1(base_image_path, replace_image_path, output_path, target_area, target_text, fused_path, location_prompt=None, scale_factor=1.2):
    try:
        base_img_tensor = load_and_transform(base_image_path, t_size)
        base_pil_image = Image.open(base_image_path).convert("RGB")
        replace_pil_image = Image.open(replace_image_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error: Image file not found. Check asset paths.")
        return

    manager = ModelManager(device=device)
    rae = manager.load_rae()
    cleanup_memory()

    fused_features_seq, fused_features_2d = stage1_extract_features(
        rae, base_img_tensor, base_pil_image, replace_pil_image, 
        scale_factor=scale_factor, target_area=target_area, 
        target_text=target_text, output_path=output_path, 
        location_prompt=location_prompt,
        use_smart_placement=True
    )

    print(f"Intermediate fusion check saved to: {output_path}")

    torch.save({
        'fused_features': fused_features_2d.cpu(),
        'target_area': target_area,
        'target_text': target_text,
        'base_image': base_image_path,
        'replace_image': replace_image_path
    }, fused_path)

    del rae
    cleanup_memory()


if __name__ == "__main__":
    base_image_path = "assets/group5/base.png"
    replace_image_path = "assets/group5/r2.png"
    output_path = "assets/group5/stage1_result/"
    target_area = "bottom_right"
    target_text = "rabbit"
    scale_factor = 0.5 # Slightly Adjusted
    location_prompt = "grass foreground"

    os.makedirs(output_path, exist_ok=True)
    fused_path = os.path.join(output_path, "fused_results.pt")

    stage1(base_image_path, replace_image_path, output_path, target_area, target_text, fused_path, location_prompt=location_prompt,scale_factor=scale_factor)
    print("Stage 1 completed.\n")