import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import sys
from typing import Tuple
import numpy as np
import datetime
import os
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.stage1.rae import RAE
from src.stage2.models.DDT import DiTwDDTHead
from torchvision.utils import save_image
from src.stage2.transport import create_transport, Sampler
import src.semantic.segmentation as clip
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


def get_cropped_object_tensor(
        raw_image: Image.Image,
        target_text: str,
        scale_factor: float = 1.2,
        background_mode: str = "mean",
        target_size_for_encoder: int = 448
) -> Tuple[torch.Tensor, torch.Tensor]:
    W_orig, H_orig = raw_image.size

    # Generate Mask via CLIPSeg
    mask = clip.extract_semantic_mask_with_clipseg(
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
        mask_to_scale = Image.new("L", (target_size_for_encoder, target_size_for_encoder), 255)
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


def hard_semantic_fusion(
        base_latent: torch.Tensor,
        object_latent: torch.Tensor,
        object_mask: torch.Tensor,
        target_size: int = 16,
        target_area: str = 'bottom_right'
) -> torch.Tensor:
    B, C, H, W = base_latent.shape

    # 1. Find valid bounding box of the object in the mask (crop out extra background)
    # This step is to extract only the "dog/cat" itself, without bringing in the original background features
    mask_2d = F.interpolate(object_mask, size=(H, W), mode='bilinear')
    mask_np = mask_2d.squeeze().cpu().numpy()
    rows, cols = np.where(mask_np > 0.1)

    if len(rows) == 0:
        print("[Warning] Mask is empty in Latent Space.")
        return base_latent

    rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()

    # 2. Extract valid region features (preserve aspect ratio)
    obj_feat_crop = object_latent[:, :, rmin:rmax + 1, cmin:cmax + 1]
    obj_mask_crop = mask_2d[:, :, rmin:rmax + 1, cmin:cmax + 1]

    # 3. Calculate scaling ratio: limit the longer side to target_size
    h_orig, w_orig = obj_feat_crop.shape[2], obj_feat_crop.shape[3]
    scale = target_size / max(h_orig, w_orig)
    h_new, w_new = max(1, int(h_orig * scale)), max(1, int(w_orig * scale))

    # 4. Perform undistorted resampling (Core of M2 Scaling)
    target_object_feat = F.interpolate(obj_feat_crop, size=(h_new, w_new), mode='bicubic')
    target_mask = F.interpolate(obj_mask_crop, size=(h_new, w_new), mode='nearest')

    # 5. Calculate dynamic placement coordinates
    ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, h_new, w_new, target_area)

    # 6. Hard alignment coverage
    fused_latent = base_latent.clone()
    # Extract background patch of the same size
    base_patch = fused_latent[:, :, ts_h:te_h, ts_w:te_w]

    # Error tolerance: ensure interpolated size matches exactly
    target_object_feat = F.interpolate(target_object_feat, size=base_patch.shape[2:])
    target_mask = F.interpolate(target_mask, size=base_patch.shape[2:])

    # Blending: object + original background in the patch
    # This ensures background outside object edges is not disturbed, reducing "ghosting"
    fused_patch = target_mask * target_object_feat + (1 - target_mask) * base_patch
    fused_latent[:, :, ts_h:te_h, ts_w:te_w] = fused_patch

    return fused_latent


@torch.inference_mode()
def stage1_extract_features(rae, base_img_tensor, replace_pil_image, target_area, target_text, output_path):
    """
    Stage 1 Optimized: Accurate CLIPSeg extraction + Latent Hard-Fusion with AMP
    """
    print(f"[Stage 1] Using CLIPSeg to extract '{target_text}'...")

    # Ensure input is on device and correct dtype
    base_img_tensor = base_img_tensor.to(device, dtype=DTYPE_MODERN)

    # 1. Image-level cropping (performed on CPU/Float32 for stability)
    object_tensor, object_mask = get_cropped_object_tensor(
        raw_image=replace_pil_image,
        target_text=target_text,
        scale_factor=1.2,
        target_size_for_encoder=512  # Scaled to 512
    )
    object_tensor = object_tensor.to(device, dtype=DTYPE_MODERN)
    object_mask = object_mask.to(device, dtype=DTYPE_MODERN)

    # Debug saving
    debug_dir = os.path.join(output_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    save_image(object_tensor.float(), os.path.join(debug_dir, f"crop_obj_{timestamp}.png"))

    # 2. Encode and Fuse with Automatic Mixed Precision
    with torch.autocast(device_type='cuda', dtype=DTYPE_MODERN):
        base_latent = rae.encode(base_img_tensor)
        object_latent = rae.encode(object_tensor)

        # Using the dynamic scaling fusion to prevent stretching
        fused_latent_2d = hard_semantic_fusion(
            base_latent=base_latent,
            object_latent=object_latent,
            object_mask=object_mask,
            target_size=16,  # Adjust based on 32x32 latent grid
            target_area=target_area
        )

        # 4. Convert back to sequence format (Transformer: B, N, C)
        fused_features_seq = fused_latent_2d.flatten(2).transpose(1, 2).contiguous()

        # 5. Preview check - Decode result
        check_img = rae.decode(fused_latent_2d)
        save_image(check_img.float(), os.path.join(debug_dir, f"latent_fusion_check_{timestamp}.png"))
        print(f"Fusion preview saved to debug directory.")

    return fused_features_seq, fused_latent_2d


def create_mock_transport_and_sampler(num_steps, time_dist_shift):
    transport_config = {
        'path_type': 'Linear',
        'prediction': 'velocity',
        'time_dist_type': 'uniform',
        'time_dist_shift': time_dist_shift,
    }

    transport = create_transport(**transport_config)
    sampler_instance = Sampler(transport)
    sampler_params = {'sampling_method': 'euler', 'num_steps': num_steps}
    sample_fn = sampler_instance.sample_ode(**sampler_params)

    return sample_fn


def stage1(base_image_path, replace_image_path, output_path, target_area, target_text, fused_path):
    try:
        base_img_tensor = load_and_transform(base_image_path, 448)
        replace_pil_image = Image.open(replace_image_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error: Image file not found. Check asset paths.")
        return

    manager = ModelManager(device=device)
    rae = manager.load_rae()
    cleanup_memory()

    fused_features_seq, fused_features_2d = stage1_extract_features(
        rae, base_img_tensor, replace_pil_image, target_area, target_text, output_path
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
    manager.cleanup()


def stage2(output_path, fused_path):
    data = torch.load(fused_path, map_location=device)
    base_fused_features_2d = data['fused_features'].to(device, dtype=DTYPE_MODERN)
    B, C, H, W = base_fused_features_2d.shape

    # editing_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
    editing_strengths = [0.8]
    generated_images_list = []
    num_steps = 10
    sample_fn = create_mock_transport_and_sampler(num_steps, time_dist_shift=1.0)

    manager = ModelManager(device=device)

    # 1. Load DiT to GPU
    dit_model = manager.load_dit()
    model_fwd = dit_model.forward
    dummy_y = torch.zeros(B, dtype=torch.long, device=device)
    model_kwargs = dict(y=dummy_y, s=None, mask=None)

    # 2. Keep RAE on CPU initially to save VRAM
    rae = manager.load_rae()
    rae.cpu()

    print(f"Running Stage 2 Optimized (Steps: {num_steps})")
    initial_noise = torch.randn((B, C, H, W), device=device, dtype=DTYPE_MODERN)

    for t in editing_strengths:
        noise_scale = 0.8 if t >= 0.7 else 0.5

        with torch.autocast(device_type='cuda', dtype=DTYPE_MODERN):
            initial_latents = (1.0 - noise_scale * t) * base_fused_features_2d + noise_scale * t * initial_noise

            # Sampling with DiT
            samples_sequence = sample_fn(initial_latents, model_fwd, **model_kwargs)
            sampled_latents = samples_sequence[-1]

        # 3. Offload DiT and bring RAE to GPU for decoding
        dit_model.cpu()
        rae.to(device)
        cleanup_memory()

        if sampled_latents.dim() == 5:
            sampled_latents = sampled_latents.squeeze(1)

        # Decode
        with torch.autocast(device_type='cuda', dtype=DTYPE_MODERN):
            generated_image = rae.decode(sampled_latents)
            generated_images_list.append(generated_image.cpu())
        # 4. Bring DiT back if there were more iterations (optional)
        # dit_model.to(device)

        cleanup_memory()

    if generated_images_list:
        stitched_image = torch.cat(generated_images_list, dim=2)
        save_image(stitched_image, f"{output_path}stitched_{timestamp}.png")

    manager.cleanup()


if __name__ == "__main__":
    # base_image_path = "assets/group5/2.png"
    # replace_image_path = "assets/group5/1.png"
    # output_path = "assets/group5/"
    # target_area = "bottom_right"
    # target_text = "animal"

    base_image_path = "assets/group1/background.png"
    replace_image_path = "assets/group1/replace.png"
    output_path = "assets/group1/"
    target_area = "bottom_left"
    target_text = "animal"

    # base_image_path = "assets/group2/2.png"
    # replace_image_path = "assets/group2/1.png"
    # output_path = "assets/group2/"
    # target_area = "bottom_right"
    # target_text = "animal"

    # Run stage1 only first to check the fusion
    fused_path = os.path.join(output_path, "fused_results.pt")
    # stage1(base_image_path, replace_image_path, output_path, target_area, target_text, fused_path)

    cleanup_memory()

    # Uncomment to run stage2
    stage2(output_path, fused_path)

