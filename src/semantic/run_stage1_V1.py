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
from src.semantic.resize import apply_m1_scaling, adaptive_target_size, apply_m2_latent_scaling
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
        scale_factor: float = 1.2,
        background_mode: str = "mean",
        target_size_for_encoder: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    # Apply M1 Scaling to both object and mask
    final_obj_pil = apply_m1_scaling(obj_to_scale, scale_factor, target_size_for_encoder, background_mode)

    # For mask, we use black background (0) regardless of mode
    final_mask_pil = apply_m1_scaling(mask_to_scale, scale_factor, target_size_for_encoder, background_mode="black")

    # Convert to Tensors
    obj_tensor = T.ToTensor()(final_obj_pil).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = T.ToTensor()(final_mask_pil).unsqueeze(0).to(device, dtype=dtype)

    if mask_tensor.shape[1] > 1:
        mask_tensor = mask_tensor[:, 0:1, :, :]

    return obj_tensor, (mask_tensor > 0).float()


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
    
    # Decoupled scaling: 1.0 means the object occupies 60% of background's short side
    base_unit = min(H, W) * 0.6
    target_size = max(min_size, int(base_unit * scale_factor))
    target_size = min(target_size, min(H, W) - 2)

    # Use bilinear for Soft-Mask to get smooth edges
    mask_latent = F.interpolate(object_mask, size=(H, W), mode="bilinear", align_corners=False)
    
    # Find the bounding box in latent space
    mask_np = (mask_latent[0, 0] > 0.3).cpu().numpy()
    ys, xs = np.where(mask_np)
    if len(xs) == 0: 
        return base_latent

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    obj_crop = object_latent[:, :, y0:y1 + 1, x0:x1 + 1]
    mask_crop = mask_latent[:, :, y0:y1 + 1, x0:x1 + 1]

    # Calculate scale for latent patches
    h0, w0 = obj_crop.shape[2:]
    scale = target_size / max(h0, w0)
    h1, w1 = max(1, int(h0 * scale)), max(1, int(w0 * scale))

    obj_rs = F.interpolate(obj_crop, (h1, w1), mode="bicubic")
    mask_rs = F.interpolate(mask_crop, (h1, w1), mode="bilinear", align_corners=False)

    # Determine coordinates
    effective_smart = False
    if use_smart_placement and location_prompt and raw_base_image:
        coords = get_text_guided_coords(raw_base_image, location_prompt, h1, w1, (H, W))
        if coords:
            ts_h, te_h, ts_w, te_w = coords
            effective_smart = True
    
    if not effective_smart:
        ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, h1, w1, target_area)

    # Execute Soft-Fusion (Alpha Blending)
    fused = base_latent.clone()
    base_patch = base_latent[:, :, ts_h:te_h, ts_w:te_w]
    
    # Ensure dimensions match perfectly
    obj_rs = F.interpolate(obj_rs, base_patch.shape[2:], mode="bicubic")
    mask_rs = F.interpolate(mask_rs, base_patch.shape[2:], mode="bilinear", align_corners=False)

    # Smooth blending reduces edge artifacts
    fused[:, :, ts_h:te_h, ts_w:te_w] = (
        mask_rs * obj_rs + (1.0 - mask_rs) * base_patch
    )
    return fused


@torch.inference_mode()
def stage1_extract_features(rae, base_img_tensor, base_pil_image, replace_pil_image, 
                           scale_factor, target_area, target_text, output_path,
                           location_prompt=None, use_smart_placement=True):
    # M1 Phase: Extract high-quality features at fixed scale (1.0)
    object_tensor, object_mask = get_cropped_object_tensor(
        raw_image=replace_pil_image,
        target_text=target_text,
        scale_factor=1.0, 
        target_size_for_encoder=t_size
    )
    object_tensor = object_tensor.to(device, dtype=DTYPE_MODERN)
    object_mask = object_mask.to(device, dtype=DTYPE_MODERN)

    save_image(object_tensor.float(), os.path.join(output_path, f"crop_obj_{timestamp}.png"))

    with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
        base_latent = rae.encode(base_img_tensor)
        object_latent = rae.encode(object_tensor)

        # M2 Phase: Apply final scale_factor only in Latent Space
        fused_latent_2d = semantic_fusion(
            base_latent,
            object_latent,
            object_mask,
            raw_base_image=base_pil_image,
            location_prompt=location_prompt,
            target_area=target_area,
            scale_factor=scale_factor,
            min_size=8,
            use_smart_placement=use_smart_placement
        )
        
        fused_features_seq = fused_latent_2d.flatten(2).transpose(1, 2).contiguous()
        
        # Save preview
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

    with torch.no_grad():
        intermediate_img = rae.decode(fused_features_2d)
        check_path = os.path.join(output_path, f"fusion_check_stage1_{timestamp}.png")
        save_image(intermediate_img, check_path)
        print(f"Intermediate fusion check saved to: {check_path}")

    torch.save({
        'fused_features': fused_features_2d.cpu(),
        'target_area': target_area,
        'target_text': target_text,
        'base_image': base_image_path,
        'replace_image': replace_image_path
    }, fused_path)

    print(f"Saved fused features to: {fused_path}")

    del rae
    cleanup_memory()


if __name__ == "__main__":
    # base_image_path = "assets/group0/base1.png"
    # replace_image_path = "assets/group0/r2.png"
    # output_path = "assets/group0/"
    # target_area = "center"
    # target_text = "orange"
    # scale_factor = 0.1
    # location_prompt = "table"

    # base_image_path = "assets/group1/base.png"
    # replace_image_path = "assets/group1/r.png"
    # output_path = "assets/group1/stage1_result/"
    # target_area = "bottom_right"
    # target_text = "dog"
    # scale_factor = 1.0
    # location_prompt = "grass central"

    base_image_path = "assets/group3/base.png"
    replace_image_path = "assets/group3/r1.png"
    output_path = "assets/group3/stage1_result/"
    target_area = "top_right"
    target_text = "eagel"
    scale_factor = 1.0
    location_prompt = "sky central"

    # base_image_path = "assets/group4/base.png"
    # replace_image_path = "assets/group4/r.png"
    # output_path = "assets/group4/stage1_result/"
    # target_area = "top_right"
    # target_text = "butterfly"
    # scale_factor = 1.0
    # location_prompt = "on flowers"

    # base_image_path = "assets/group5/base.png"
    # replace_image_path = "assets/group5/r1.png"
    # output_path = "assets/group5/stage1_result/"
    # target_area = "bottom_right"
    # target_text = "animal"
    # scale_factor = 1.0
    # location_prompt = "grass foreground central"

    # base_image_path = "assets/group7/base.png"
    # replace_image_path = "assets/group7/r1.png"
    # output_path = "assets/group7/stage1_result/"
    # target_area = "top_right"
    # target_text = "fish"
    # scale_factor = 1.0
    # location_prompt = "central"

    # base_image_path = "assets/group10/base1.png"
    # replace_image_path = "assets/group10/r1.png"
    # output_path = "assets/group10/stage1_result/"
    # target_area = "top_right"
    # target_text = "eagle"
    # scale_factor = 1.0
    # location_prompt = "top_right"

    os.makedirs(output_path, exist_ok=True)
    fused_path = os.path.join(output_path, "fused_results.pt")


    stage1(base_image_path, replace_image_path, output_path, target_area, target_text, fused_path, location_prompt=location_prompt,scale_factor=scale_factor)
    print("Stage 1 completed.\n")