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
from src.semantic.resize import apply_m1_scaling
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE, DTYPE_MODERN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
t_size = 448

def get_cropped_object_tensor(
        raw_image: Image.Image,
        target_text: str,
        scale_factor: float = 1.0,
        background_mode: str = "mean",
        target_size_for_encoder: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the object and applies M1 (Pixel) scaling.
    For this optimized pipeline, we usually keep scale_factor = 1.0 here 
    to ensure the encoder gets the best possible quality.
    """
    W_orig, H_orig = raw_image.size

    # Generate Mask via CLIPSeg
    mask = extract_semantic_mask_with_clipseg(
        image=raw_image,
        target_text=target_text,
        feature_size=(H_orig, W_orig),
        threshold=0.3
    ).cpu().squeeze()

    binary_mask = (mask.numpy() > 0.3).astype(np.uint8)
    rows, cols = np.any(binary_mask, axis=1), np.any(binary_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        print(f"[Warning] No object '{target_text}' found. Using full image.")
        obj_to_scale = raw_image
        mask_to_scale = Image.new("L", (t_size, t_size), 255)
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Tight crop of the object
        obj_to_scale = Image.fromarray(np.array(raw_image)[rmin:rmax, cmin:cmax])
        mask_to_scale = Image.fromarray((binary_mask[rmin:rmax, cmin:cmax] * 255).astype(np.uint8), mode='L')

    # Apply M1 Scaling
    final_obj_pil = apply_m1_scaling(obj_to_scale, scale_factor, target_size_for_encoder, background_mode)
    
    # For mask, use black background
    final_mask_pil = apply_m1_scaling(mask_to_scale, scale_factor, target_size_for_encoder, background_mode="black")

    # Convert to Tensors
    obj_tensor = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device, dtype=dtype)

    if mask_tensor.shape[1] > 1:
        mask_tensor = mask_tensor[:, 0:1, :, :]

    return obj_tensor, (mask_tensor > 0).float()


def semantic_fusion_m2(
    base_latent: torch.Tensor,
    object_latent: torch.Tensor,
    object_mask: torch.Tensor,
    raw_base_image: Image.Image,
    location_prompt: str,
    target_area: str,
    scale_factor: float,
    min_size: int = 8,
    use_smart_placement: bool = True
) -> torch.Tensor:
    """
    Optimized Fusion Function handling M2 Latent Scaling.
    
    1. Calculates target size based on base image dimensions and scale_factor.
    2. Resizes the object latent features and mask.
    3. Fuses them into the base latent.
    """
    B, C, H, W = base_latent.shape
    
    # --- M2 Core Logic: Calculate Target Size ---
    # base_unit is defined as 60% of the background's shortest side.
    # scale_factor directly multiplies this unit.
    base_unit = min(H, W) * 0.6 
    target_long_edge = int(base_unit * scale_factor)
    
    # Constraint limits to ensure it fits and isn't too small
    target_long_edge = max(min_size, min(target_long_edge, min(H, W) - 2))

    # 1. Prepare Mask (interpolate to full size first to find ROI)
    mask_latent = F.interpolate(object_mask, size=(H, W), mode="bilinear", align_corners=False)
    
    # 2. Find ROI (Region of Interest)
    mask_np = (mask_latent[0, 0] > 0.3).cpu().numpy()
    ys, xs = np.where(mask_np)
    if len(xs) == 0: 
        return base_latent # Return original if object is empty

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    
    # Extract the high-quality latent crop (from M1=1.0 encoding)
    obj_crop = object_latent[:, :, y0:y1 + 1, x0:x1 + 1]
    mask_crop = mask_latent[:, :, y0:y1 + 1, x0:x1 + 1]

    # --- M2 Scaling: Resize in Latent Space ---
    h_old, w_old = obj_crop.shape[2:]
    
    # Calculate resizing ratio
    scale_ratio = target_long_edge / max(h_old, w_old)
    
    h_new = max(1, int(h_old * scale_ratio))
    w_new = max(1, int(w_old * scale_ratio))

    # Resize: Bicubic for features (smoothness), Bilinear for mask
    obj_rs = F.interpolate(obj_crop, size=(h_new, w_new), mode="bicubic", align_corners=False)
    mask_rs = F.interpolate(mask_crop, size=(h_new, w_new), mode="bilinear", align_corners=False)

    # 3. Determine Coordinates
    effective_smart = False
    if use_smart_placement and location_prompt and raw_base_image:
        coords = get_text_guided_coords(raw_base_image, location_prompt, h_new, w_new, (H, W))
        if coords:
            ts_h, te_h, ts_w, te_w = coords
            effective_smart = True
    
    if not effective_smart:
        ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, h_new, w_new, target_area)

    # 4. Fusion
    fused = base_latent.clone()
    base_patch = base_latent[:, :, ts_h:te_h, ts_w:te_w]
    
    # Double check dimensions for rounding errors
    if base_patch.shape[2:] != obj_rs.shape[2:]:
        obj_rs = F.interpolate(obj_rs, base_patch.shape[2:], mode="bicubic", align_corners=False)
        mask_rs = F.interpolate(mask_rs, base_patch.shape[2:], mode="bilinear", align_corners=False)

    # Alpha Blending
    fused[:, :, ts_h:te_h, ts_w:te_w] = (
        mask_rs * obj_rs + (1.0 - mask_rs) * base_patch
    )
    return fused


def stage1_batch_optimized(
    base_image_path, 
    replace_image_path, 
    output_path, 
    target_area, 
    target_text, 
    location_prompt=None, 
    scale_factors=[1.0]
):
    try:
        base_img_tensor = load_and_transform(base_image_path, t_size)
        base_pil_image = Image.open(base_image_path).convert("RGB")
        replace_pil_image = Image.open(replace_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found.")
        return

    # 1. Load Model
    manager = ModelManager(device=device)
    rae = manager.load_rae()
    cleanup_memory()

    print("--- Pre-calculation: Encoding Base and Object (Fixed M1=1.0) ---")
    
    # 2. Pre-calculation: Encode ONLY ONCE!
    # Force M1 scale = 1.0 (Ensures max feature details)
    object_tensor, object_mask = get_cropped_object_tensor(
        raw_image=replace_pil_image,
        target_text=target_text,
        scale_factor=1.0,  # <--- FIXED at 1.0
        target_size_for_encoder=t_size
    )
    object_tensor = object_tensor.to(device, dtype=DTYPE_MODERN)
    object_mask = object_mask.to(device, dtype=DTYPE_MODERN)

    with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
        # Get Latent Features
        base_latent = rae.encode(base_img_tensor)
        object_latent = rae.encode(object_tensor)

    # 3. Loop Generation: Vary M2 Scale only
    for scale in scale_factors:
        print(f"--- Generating M2 Scale: {scale} ---")
        
        with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
            fused_latent_2d = semantic_fusion_m2(
                base_latent,
                object_latent,
                object_mask,
                raw_base_image=base_pil_image,
                location_prompt=location_prompt,
                target_area=target_area,
                scale_factor=scale, # <--- Dynamic M2 scale
                use_smart_placement=True
            )
            
            # Save Check Image (Preview)
            check_img = rae.decode(fused_latent_2d)
            check_filename = f"fusion_check_scale_{scale}_{timestamp}.png"
            save_image(check_img.float(), os.path.join(output_path, check_filename))

            # Save .pt Data
            fused_filename = f"fused_results_scale_{scale}.pt"
            torch.save({
                'fused_features': fused_latent_2d.cpu(),
                'target_area': target_area,
                'target_text': target_text,
                'scale_factor': scale,
                'timestamp': timestamp
            }, os.path.join(output_path, fused_filename))
            
            print(f"Saved: {fused_filename}")

    print("Batch processing completed.")
    del rae
    cleanup_memory()


if __name__ == "__main__":
    # Path Configuration
    # base_image_path = "assets/group2/base1.png"
    # replace_image_path = "assets/group2/r2.png"
    # output_path = "assets/group2/stage1_result/"
    
    # target_area = "center"
    # target_text = "orange"
    # location_prompt = "on the table"

    # base_image_path = "assets/group1/base.png"
    # replace_image_path = "assets/group1/r.png"
    # output_path = "assets/group1/stage1_result/"
    # target_area = "bottom_right"
    # target_text = "dog"
    # location_prompt = "grass central"

    # base_image_path = "assets/group5/base.png"
    # replace_image_path = "assets/group5/r1.png"
    # output_path = "assets/group5/stage1_result/"
    # target_area = "bottom_right"
    # target_text = "animal"
    # location_prompt = "grass foreground central"

    base_image_path = "assets/group4/base.png"
    replace_image_path = "assets/group4/r.png"
    output_path = "assets/group4/stage1_result3/"
    target_area = "top_right"
    target_text = "butterfly"
    scale_factor = 1.0
    location_prompt = "on flowers"
    
    # List of M2 Scales to generate
    # 0.5 = Shrink to half relative size
    # 1.0 = Standard size (approx 60% of background short side)
    # 1.5 = Enlarge to 1.5x relative size
    scale_factors = [0.2, 0.4, 0.6, 0.8, 1.2]

    os.makedirs(output_path, exist_ok=True)

    stage1_batch_optimized(
        base_image_path, 
        replace_image_path, 
        output_path, 
        target_area, 
        target_text, 
        location_prompt=location_prompt,
        scale_factors=scale_factors
    )