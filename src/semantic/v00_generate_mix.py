import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import Tuple, Dict, Callable, Optional
from math import sqrt
import numpy as np
import gc
import datetime
import time
from pathlib import Path
from src.stage1.rae import RAE
from src.stage2.models.DDT import DiTwDDTHead
from torchvision.utils import save_image
from src.stage2.transport import create_transport, Sampler

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
print(f"Using device: {device} | dtype: {dtype}")

H_PATCH = 16
W_PATCH = 16

FusionResult = Tuple[torch.Tensor, int, int]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def load_and_transform(path: str, target_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def extract_semantic_mask(rae_instance: 'RAE', raw_image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = raw_image_tensor.clone()
        _, _, h, w = x.shape
        if h != rae_instance.encoder_input_size or w != rae_instance.encoder_input_size:
            x = F.interpolate(x, size=(rae_instance.encoder_input_size, rae_instance.encoder_input_size),
                              mode='bicubic', align_corners=False)
        x = (x - rae_instance.encoder_mean.to(x.device)) / rae_instance.encoder_std.to(x.device)

        attn_weights = rae_instance.encoder.get_attention_maps(x)
        B, num_heads, N = attn_weights.shape
        H = W = int(sqrt(N))

        avg_attn = attn_weights.mean(dim=1)
        attn_2d = avg_attn.view(B, H, W)

        threshold_value = torch.quantile(attn_2d.flatten(start_dim=1), 0.75,
                                         dim=1, keepdim=True).view(B, 1, 1).to(attn_2d.device)
        mask = (attn_2d > threshold_value).float().unsqueeze(1)

        return mask.to(raw_image_tensor.device)


def _calculate_target_coords(H: int, W: int, size: int, area: str) -> Tuple[int, int, int, int]:
    """Calculate start and end coordinates for replacement based on area specification."""
    if area == 'top_left':
        return 0, size, 0, size
    elif area == 'top_right':
        return 0, size, W - size, W
    elif area == 'bottom_left':
        return H - size, H, 0, size
    elif area == 'center':
        start_H = (H - size) // 2
        start_W = (W - size) // 2
        return start_H, start_H + size, start_W, start_W + size
    elif area == 'bottom_right':
        return H - size, H, W - size, W
    else:
        print(f"[Warning] Unknown target_area '{area}'. Using {target_area}.")
        return H - size, H, W - size, W


def semantic_placement_fusion(
        base_feat: torch.Tensor,
        content_feat: torch.Tensor,
        replace_raw_tensor: torch.Tensor,
        rae_instance: 'RAE',
        extract_area: str = 'center',
        target_size: int = 8,
        target_area: str = 'bottom_right'
) -> Tuple[torch.Tensor, int, int]:
    """
    1. Extract semantic subject from content features using DINOv2 attention
    2. Crop source feature map based on extract_area parameter
    3. Replace target_area in base features with cropped content
    """
    B, N, C = base_feat.shape
    H = W = int(math.sqrt(N))

    base_2d = base_feat.transpose(1, 2).view(B, C, H, W).clone()
    content_2d = content_feat.transpose(1, 2).view(B, C, H, W)

    replace_mask = extract_semantic_mask(rae_instance, replace_raw_tensor)
    replace_foreground_2d = content_2d * replace_mask

    start_replace_H, end_replace_H, start_replace_W, end_replace_W = _calculate_target_coords(
        H=H, W=W, size=target_size, area=extract_area
    )

    cropped_replace_feat = replace_foreground_2d[:, :,
    start_replace_H:end_replace_H,
    start_replace_W:end_replace_W]

    start_H_base, end_H_base, start_W_base, end_W_base = _calculate_target_coords(
        H=H, W=W, size=target_size, area=target_area
    )

    print(f"Extraction area: '{extract_area}' ({start_replace_H}:{end_replace_H}, {start_replace_W}:{end_replace_W})")
    print(f"Target replacement area: '{target_area}' ({start_H_base}:{end_H_base}, {start_W_base}:{end_W_base})")

    fused_2d = base_2d.clone()
    fused_2d[:, :, start_H_base:end_H_base, start_W_base:end_W_base] = cropped_replace_feat

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


def stage1_extract_features(rae, base_img_tensor, replace_img_tensor, target_area, extract_area):
    with torch.no_grad():
        base_features_seq = rae.encode(base_img_tensor).contiguous().view(1, 768, -1).transpose(1, 2)
        replace_features_seq = rae.encode(replace_img_tensor).contiguous().view(1, 768, -1).transpose(1, 2)
        B, N, C = base_features_seq.shape
    print("\n--- Running DINOv2 Semantic Placement Fusion: replace Foreground to base Right-Bottom ---")

    fused_features_seq, H_fused, W_fused = semantic_placement_fusion(
        base_feat=base_features_seq,
        content_feat=replace_features_seq,
        replace_raw_tensor=replace_img_tensor,
        rae_instance=rae,
        target_size=8,
        target_area=target_area
    )
    print("fused_features_seq.shape")
    print(fused_features_seq.shape)
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

    # Instantiate Transport
    # Note: This needs to be adjusted based on your actual create_transport function
    transport = create_transport(
        **transport_config,
    )

    # Instantiate Sampler
    # Sampler class usually takes a transport object in __init__
    sampler_instance = Sampler(transport)

    sampler_params = {'sampling_method': 'euler', 'num_steps': num_steps}

    # sample_fn contains the denoising loop logic
    sample_fn = sampler_instance.sample_ode(**sampler_params)

    return sample_fn


class EDMNoiseScheduler:
    """EDM Noise Scheduler"""

    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, num_steps: int) -> torch.Tensor:
        """Get sigma values for each timestep"""
        step_indices = torch.arange(num_steps, device=device)
        t = (step_indices / (num_steps - 1)).to(device)

        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)

        sigmas = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** self.rho
        return sigmas


def add_noise_to_latents(latents: torch.Tensor, t: float, scheduler: EDMNoiseScheduler) -> torch.Tensor:
    """Add noise to latents at specific timestep t (0-1)"""
    # Get sigma value for timestep t
    sigma_t = (scheduler.sigma_max ** (1 / scheduler.rho) +
               t * (scheduler.sigma_min ** (1 / scheduler.rho) - scheduler.sigma_max ** (
                    1 / scheduler.rho))) ** scheduler.rho

    # Generate noise
    noise = torch.randn_like(latents)

    # Add noise to latents
    noisy_latents = latents + noise * sigma_t

    return noisy_latents


def stage1(base_image_path, replace_image_path, output_path, target_area, extract_area,fused_path):
    try:
        base_img_tensor = load_and_transform(base_image_path, 224)
        replace_img_tensor = load_and_transform(replace_image_path, 224)
    except FileNotFoundError as e:
        print(f"Error: Image file not found. Check asset paths.")
        return

    decoder_weights_path = 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt'
    if not os.path.exists(decoder_weights_path):
        print(f"Error: Decoder weights file not found at {decoder_weights_path}")
        return
    rae = load_rae_model(decoder_weights_path)
    cleanup_memory()

    fused_features_seq, fused_features_2d = stage1_extract_features(rae, base_img_tensor, replace_img_tensor,
                                                                    target_area, extract_area)
    print(f"Fused features seq shape: {fused_features_seq.shape}")
    print(f"Fused features 2D shape: {fused_features_2d.shape}")

    B, C, H, W = fused_features_2d.shape

    
    torch.save({
        'fused_features': fused_features_2d.cpu(),
        'target_area': target_area,
        'extract_area': extract_area,
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
    editing_strengths = [0.8] 
    generated_images_list = []
    # Common model/sampler setup (outside the loop)
    num_steps = 10 # Increased steps for better quality
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
        t_str = f"{t:.2f}".replace('.', 'p') # 例如 0.35 -> 0p35
        t_filename = os.path.join(output_path, f"output_t_{t_str}_{timestamp}.png")
        save_image(generated_image[0].cpu(), t_filename)
        
        print(f"-> Saved: {t_filename}")
        saved_files_count += 1
        
        cleanup_memory()

    del base_fused_features_2d
    cleanup_memory()

    
    if generated_images_list:
        # Concatenate tensors along the width dimension (dim=2 for C,H,W or dim=3 for B,C,H,W)
        # Since we collected C, H, W tensors (B=1 was squeezed out), we use dim=2
        stitched_image = torch.cat(generated_images_list, dim=2)

        # --- 3. Save the final output ---
        output_filename = f"{output_path}stitched_output_t{editing_strengths[0]:.1f}-{editing_strengths[-1]:.1f}_{timestamp}.png"

        # save_image handles C, H, W tensor
        save_image(stitched_image, output_filename)

        print(f"\n--- Final Output ---")
        print(f"Successfully stitched and saved all {len(editing_strengths)} levels.")
        print(f"Saved to: {output_filename}. (Total Size: {stitched_image.shape[1]}x{stitched_image.shape[2]})")
    else:
        print("\nError: No images were generated to stitch.")

    
    cleanup_memory()


def cleanup_memory():
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # base_image_path = "assets/group1/background.png"
    # replace_image_path = "assets/group1/replace.png"
    # output_path = "assets/group1/"
    # target_area = "bottom_left"
    # extract_area = "center"

    # base_image_path = "assets/group2/2.png"
    # replace_image_path = "assets/group2/3.png"
    # output_path = "assets/group2"
    # target_area = "bottom_right"
    # extract_area = "center"

    # base_image_path = "assets/group3/3.png"
    # replace_image_path = "assets/group3/1.png"
    # output_path = "assets/group3"
    # target_area = "bottom_right"
    # extract_area = "center"
    
    # base_image_path = "assets/group4/1.png"
    # replace_image_path = "assets/group4/2.png"
    # output_path = "assets/group4/"
    # target_area = "top_left"
    # extract_area = "center"

    # base_image_path = "assets/group5/1.png"
    # replace_image_path = "assets/group5/2.png"
    # output_path = "assets/group5/"
    # target_area = "bottom_right"
    # extract_area = "center"

    # base_image_path = "assets/group6/1.png"
    # replace_image_path = "assets/group6/2.png"
    # output_path = "assets/group6/"
    # target_area = "bottom_right"
    # extract_area = "center"

    # base_image_path = "assets/group7/1.png"
    # replace_image_path = "assets/group7/2.png"
    # output_path = "assets/group7/"
    # target_area = "bottom_right"
    # extract_area = "center"

    # base_image_path = "assets/group8/1.png"
    # replace_image_path = "assets/group8/2.png"
    # output_path = "assets/group8/"
    # target_area = "bottom_right"
    # extract_area = "center"

    # base_image_path = "assets/group9/1.png"
    # replace_image_path = "assets/group9/2.png"
    # output_path = "assets/group9/"
    # target_area = "bottom_right"
    # extract_area = "center"

    # base_image_path = "assets/group10/1.png"
    # replace_image_path = "assets/group10/2.png"
    # output_path = "assets/group10/"
    # target_area = "top_right"
    # extract_area = "center"
    
    base_image_path = "assets/group2/2.png"
    replace_image_path = "assets/group2/1.png"
    output_path = "assets/group2/"
    target_area = "bottom_right"
    extract_area = "center"

    # stage1(base_image_path, replace_image_path, output_path, target_area, extract_area,fused_path = os.path.join(output_path,
    #                           f"fused_results.pt"))
    print("-------stage1 ending")
    cleanup_memory()
    stage2(output_path, fused_path=os.path.join(output_path,
                                                f"fused_results.pt"))
    print("-------stage2 ending")
