import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import sys
import os
from typing import Tuple, Dict, Callable, Optional
from math import sqrt
import numpy as np
import datetime
import time
from pathlib import Path
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.stage1.rae import RAE
from src.stage2.models.DDT import DiTwDDTHead
from torchvision.utils import save_image
from src.stage2.transport import create_transport, Sampler
import src.semantic.segmentation as clip
from src.semantic.utils import cleanup_memory,load_and_transform,_calculate_target_coords
# --- 全局配置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"Using device: {device} | dtype: {dtype}")

H_PATCH = 16
W_PATCH = 16

FusionResult = Tuple[torch.Tensor, int, int]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

# # --- CLIPSeg模型初始化 ---
# CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
# print(f"Loading CLIPSeg model to {device}...")
# clipseg_processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL_ID)
# clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_ID).to(device).eval()


# def extract_semantic_mask_with_clipseg(image: Image.Image, target_text: str, feature_size: Tuple[int, int], threshold: float = 0.5) -> torch.Tensor:
#     """
#     Extract semantic mask using CLIPSeg for a specific text prompt.

#     Args:
#         image: PIL Image (RGB)
#         target_text: Text description of target object
#         feature_size: Target feature map size (H, W)
#         threshold: Probability threshold for mask

#     Returns:
#         Binary mask tensor [1, 1, H, W]
#     """
#     # Preprocess image and text
#     inputs = clipseg_processor(
#         text=[target_text],
#         images=[image],
#         padding="max_length",
#         return_tensors="pt"
#     ).to(device)

#     # Get CLIPSeg predictions
#     with torch.no_grad():
#         outputs = clipseg_model(**inputs)

#     # Sigmoid activation and resize to feature size
#     preds = torch.sigmoid(outputs.logits).unsqueeze(1)  # [1, 1, 352, 352]

#     # Resize to target feature size
#     H, W = feature_size
#     mask_tensor = F.interpolate(
#         preds,
#         size=(H, W),
#         mode="bilinear",
#         align_corners=False
#     )  # [1, 1, H, W]

#     # Apply threshold to get binary mask
#     binary_mask = (mask_tensor > threshold).float()

#     return binary_mask


# def load_and_transform(path: str, target_size: int = 224) -> torch.Tensor:
#     img = Image.open(path).convert("RGB")
#     img = img.resize((target_size, target_size), Image.LANCZOS)
#     tensor = T.ToTensor()(img).unsqueeze(0)
#     tensor = tensor.to(device=device, dtype=dtype)
#     return tensor

# def _calculate_target_coords(H: int, W: int, size: int, area: str) -> Tuple[int, int, int, int]:
#     """Calculate start and end coordinates for replacement based on area specification."""
#     if area == 'top_left':
#         return 0, size, 0, size
#     elif area == 'top_right':
#         return 0, size, W - size, W
#     elif area == 'bottom_left':
#         return H - size, H, 0, size
#     elif area == 'center':
#         start_H = (H - size) // 2
#         start_W = (W - size) // 2
#         return start_H, start_H + size, start_W, start_W + size
#     elif area == 'bottom_right':
#         return H - size, H, W - size, W
#     else:
#         print(f"[Warning] Unknown target_area '{area}'. Using bottom_right.")
#         return H - size, H, W - size, W


def semantic_placement_fusion(
        base_feat: torch.Tensor,
        content_feat: torch.Tensor,
        replace_raw_image: Image.Image,  # Changed from tensor to PIL Image
        target_text: str = "object",
        target_size: int = 8,
        target_area: str = 'bottom_right'
) -> Tuple[torch.Tensor, int, int]:
    """
    Semantic placement fusion using CLIPSeg for object extraction

    1. Extract semantic subject from content features using CLIPSeg
    2. Apply mask to get foreground features
    3. Resize foreground to target size
    4. Replace target_area in base features with cropped content
    """
    B, N, C = base_feat.shape
    H = W = int(math.sqrt(N))

    base_2d = base_feat.transpose(1, 2).view(B, C, H, W).clone()
    content_2d = content_feat.transpose(1, 2).view(B, C, H, W)

    # Extract semantic mask using CLIPSeg
    print(f"Extracting semantic mask for: '{target_text}'")
    replace_mask = clip.extract_semantic_mask_with_clipseg(
        image=replace_raw_image,
        target_text=target_text,
        feature_size=(H, W),
        threshold=0.3  # Adjust threshold as needed
    ).to(content_2d.device)

    # Apply mask to get foreground features
    replace_foreground_2d = content_2d * replace_mask

    # Calculate target area coordinates
    ts_h, te_h, ts_w, te_w = _calculate_target_coords(H, W, target_size, target_area)

    # Resize foreground to target size
    cropped_object_feat = F.interpolate(
        replace_foreground_2d,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )

    # Replace target area with cropped features
    fused_2d = base_2d.clone()
    fused_2d[:, :, ts_h:te_h, ts_w:te_w] = cropped_object_feat

    fused_seq = fused_2d.view(B, C, N).transpose(1, 2).contiguous()
    return fused_seq, H, W


def load_rae_model(decoder_weights_path: str) -> 'RAE':
    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-with-registers-base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
        decoder_config_path='configs/decoder/ViTXL',
        pretrained_decoder_path=decoder_weights_path,
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path='models/stats/dinov2/wReg_base/imagenet1k/stat.pt',
        eps=1e-5,
    ).to(device).eval()
    return rae


def stage1_extract_features(rae, base_img_tensor, replace_img_tensor, replace_pil_image, target_area, target_text):
    """Stage 1: Extract and fuse features using semantic placement"""
    with torch.no_grad():
        base_features_seq = rae.encode(base_img_tensor).contiguous().view(1, 768, -1).transpose(1, 2)
        replace_features_seq = rae.encode(replace_img_tensor).contiguous().view(1, 768, -1).transpose(1, 2)
        B, N, C = base_features_seq.shape

    print(f"\n--- Running CLIPSeg Semantic Placement Fusion: '{target_text}' to base {target_area} ---")

    fused_features_seq, H_fused, W_fused = semantic_placement_fusion(
        base_feat=base_features_seq,
        content_feat=replace_features_seq,
        replace_raw_image=replace_pil_image,
        target_text=target_text,
        target_size=8,
        target_area=target_area
    )

    print(f"fused_features_seq.shape: {fused_features_seq.shape}")
    fused_features_2d = fused_features_seq.transpose(1, 2).view(B, C, H_fused, W_fused)
    return fused_features_seq, fused_features_2d


def load_dit_model(dit_ckpt_path: str):
    model_params = {
        'input_size': 16,
        'patch_size': 1,
        'in_channels': 768,
        'hidden_size': [1152, 2048],
        'depth': [28, 2],
        'num_heads': [16, 16],
        'mlp_ratio': 4.0,
        'class_dropout_prob': 0.1,
        'num_classes': 1000,
        'use_qknorm': False,
        'use_swiglu': True,
        'use_rope': True,
        'use_rmsnorm': True,
        'wo_shift': False,
        'use_pos_embed': True
    }
    print("Instantiating DiT model (DiTwDDTHead)...")
    dit_model = DiTwDDTHead(**model_params)
    dit_model.eval()
    print(f"Loading checkpoint from: {dit_ckpt_path}")

    try:
        checkpoint = torch.load(dit_ckpt_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict_clean = {k.replace('model.', ''): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = dit_model.load_state_dict(state_dict_clean, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in model state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in model state dict: {unexpected_keys}")
        print("DiT model checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {dit_ckpt_path}")
        return None
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None

    dit_model.to(device)
    print(f"DiT model moved to device: {device}")

    return dit_model


def create_mock_transport_and_sampler(device, num_steps, time_dist_shift):
    # Define Transport configuration (based on your config yaml)
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
        base_img_tensor = load_and_transform(base_image_path, 224)
        replace_img_tensor = load_and_transform(replace_image_path, 224)
        replace_pil_image = Image.open(replace_image_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error: Image file not found. Check asset paths.")
        return

    decoder_weights_path = 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt'
    if not os.path.exists(decoder_weights_path):
        print(f"Error: Decoder weights file not found at {decoder_weights_path}")
        return
    rae = load_rae_model(decoder_weights_path)
    cleanup_memory()

    fused_features_seq, fused_features_2d = stage1_extract_features(
        rae, base_img_tensor, replace_img_tensor, replace_pil_image, target_area, target_text
    )

    print(f"Fused features seq shape: {fused_features_seq.shape}")
    print(f"Fused features 2D shape: {fused_features_2d.shape}")

    B, C, H, W = fused_features_2d.shape

    torch.save({
        'fused_features': fused_features_2d.cpu(),
        'target_area': target_area,
        'target_text': target_text,
        'base_image': base_image_path,
        'replace_image': replace_image_path
    }, fused_path)

    print(f"Saved fused features to: {fused_path}")


def stage2(output_path, fused_path):
    # 1. Load fused features ONCE
    data = torch.load(fused_path, map_location=device)
    base_fused_features_2d = data['fused_features'].to(dtype=dtype)
    B, C, H, W = base_fused_features_2d.shape

    # Define five noise levels (t values) for editing
    # editing_strengths = [0.1, 0.3, 0.6, 0.8, 0.9] # 5 levels: light edit -> full generation
    editing_strengths = [0.8]  # 5 levels: light edit -> full generation
    generated_images_list = []
    # Common model/sampler setup (outside the loop)
    num_steps = 10  # Increased steps for better quality
    sample_fn = create_mock_transport_and_sampler(device, num_steps, time_dist_shift=1.0)

    dit_ckpt_path = 'models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt'
    dit_model = load_dit_model(dit_ckpt_path)
    if dit_model is None:
        print("Failed to load DiT model — aborting generation.")
        return
    else:
        print("DiT model loaded successfully for Stage 2.")

    model_fwd = dit_model.forward
    dummy_y = torch.zeros(B, dtype=torch.long, device=base_fused_features_2d.device)
    model_kwargs = dict(y=dummy_y, s=None, mask=None)

    # Load RAE decoder ONCE
    decoder_weights_path = 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt'
    if not os.path.exists(decoder_weights_path):
        print(f"Error: Decoder weights file not found at {decoder_weights_path}")
        return
    rae = load_rae_model(decoder_weights_path)

    print(f"\n--- Running Stage 2 with {len(editing_strengths)} Editing Strengths (Steps: {num_steps}) ---")
    initial_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)
    saved_files_count = 0
    for t in editing_strengths:
        initial_latents = (1.0 - t) * base_fused_features_2d + t * initial_noise

        print(f"\n--- Generating for editing_strength t = {t:.1f} ---")

        input_latents = initial_latents

        start_time = time.time()
        samples_sequence = sample_fn(input_latents, model_fwd, **model_kwargs)
        end_time = time.time()

        sampled_latents = samples_sequence[-1]
        if sampled_latents.dim() == 5 and sampled_latents.shape[1] == 1:
            sampled_latents_4d = sampled_latents.squeeze(1)
        else:
            sampled_latents_4d = sampled_latents

        with torch.no_grad():
            generated_image = rae.decode(sampled_latents_4d)

        print(f"Sampling Time: {end_time - start_time:.2f}s")

        generated_images_list.append(generated_image[0].cpu())
        t_str = f"{t:.2f}".replace('.', 'p')
        t_filename = os.path.join(output_path, f"output_t_{t_str}_{timestamp}.png")
        save_image(generated_image[0].cpu(), t_filename)

        print(f"-> Saved: {t_filename}")
        saved_files_count += 1

        cleanup_memory()

    del base_fused_features_2d
    cleanup_memory()

    if generated_images_list:
        stitched_image = torch.cat(generated_images_list, dim=2)
        output_filename = f"{output_path}stitched_output_t{editing_strengths[0]:.1f}-{editing_strengths[-1]:.1f}_{timestamp}.png"
        save_image(stitched_image, output_filename)

        print(f"\n--- Final Output ---")
        print(f"Successfully stitched and saved all {len(editing_strengths)} levels.")
        print(f"Saved to: {output_filename}. (Total Size: {stitched_image.shape[1]}x{stitched_image.shape[2]})")
    else:
        print("\nError: No images were generated to stitch.")

    cleanup_memory()





if __name__ == "__main__":
    # Example usage with CLIPSeg text prompts

    # Group 1
    base_image_path = "assets/group1/4.png"
    replace_image_path = "assets/group1/1.png"
    output_path = "assets/group1/"
    target_area = "bottom_right"
    target_text = "animal"  # Example: if image contains an animal

    # Group 2
    # base_image_path = "assets/group2/2.png"
    # replace_image_path = "assets/group2/3.png"
    # output_path = "assets/group2/"
    # target_area = "bottom_right"
    # target_text = "person"  # Example: if image contains a person

    # Group 10 (your example)
    # base_image_path = "assets/group10/1.png"
    # replace_image_path = "assets/group10/2.png"
    # output_path = "assets/group10/"
    # target_area = "top_right"
    # target_text = "bird"  # Example: if image contains a bird

    stage1(base_image_path, replace_image_path, output_path, target_area, target_text,
           fused_path=os.path.join(output_path, f"fused_results.pt"))

    cleanup_memory()

    stage2(output_path, fused_path=os.path.join(output_path, f"fused_results.pt"))