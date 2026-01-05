import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import math 
from typing import Tuple, Union
from math import sqrt
import warnings
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.semantic.config import DEVICE, DTYPE

def apply_m1_scaling(
        img: Image.Image,
        scale_factor: float,
        target_size: int = 224,
        background_mode: str = "mean"
) -> Image.Image:
    """
    M1 Method: Scaling in pixel space using PIL.
    Resizes the object and pads it into a fixed-size canvas.
    """
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h

    if aspect > 1:
        new_w = int(target_size * scale_factor)
        new_h = int(new_w / aspect)
    else:
        new_h = int(target_size * scale_factor)
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
    new_img = Image.new("RGB", (target_size, target_size), bg_color)
    upper = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    new_img.paste(img_resized, (left, upper))

    return new_img

def apply_m2_latent_scaling(
    object_feat: torch.Tensor, 
    object_mask: torch.Tensor, 
    scale_factor: float, 
    target_latent_size: int = 16, 
    mode: str = 'bilinear'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    M2 Method: Scaling in latent space using linear interpolation.
    Maintains a fixed grid size (e.g., 16x16) by padding or cropping.
    """
    B, N, C = object_feat.shape
    H_orig = W_orig = int(math.sqrt(N))
    device = object_feat.device
    dtype = object_feat.dtype

    # Convert to 2D format [B, C, H, W]
    feat_2d = object_feat.transpose(1, 2).view(B, C, H_orig, W_orig)
    
    # Calculate scaled dimensions
    scaled_h = max(1, int(H_orig * scale_factor))
    scaled_w = max(1, int(W_orig * scale_factor))
    
    # Resize features and mask
    scaled_feat = F.interpolate(feat_2d, size=(scaled_h, scaled_w), mode=mode, align_corners=False)
    mask_latent = F.interpolate(object_mask, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)

    # Create fixed-size canvas for latent features
    canvas_feat = torch.zeros((B, C, target_latent_size, target_latent_size), device=device, dtype=dtype)
    canvas_mask = torch.zeros((B, 1, target_latent_size, target_latent_size), device=device, dtype=dtype)

    # Calculate center-alignment coordinates
    h_fit = min(scaled_h, target_latent_size)
    w_fit = min(scaled_w, target_latent_size)
    
    y_start = max(0, (target_latent_size - h_fit) // 2)
    x_start = max(0, (target_latent_size - w_fit) // 2)
    
    src_y_start = max(0, (scaled_h - h_fit) // 2)
    src_x_start = max(0, (scaled_w - w_fit) // 2)

    # Paste scaled features into the center of the canvas
    canvas_feat[:, :, y_start:y_start+h_fit, x_start:x_start+w_fit] = \
        scaled_feat[:, :, src_y_start:src_y_start+h_fit, src_x_start:src_x_start+w_fit]
    
    canvas_mask[:, :, y_start:y_start+h_fit, x_start:x_start+w_fit] = \
        mask_latent[:, :, src_y_start:src_y_start+h_fit, src_x_start:src_x_start+w_fit]

    # Revert to sequence format [B, N, C]
    final_feat_seq = canvas_feat.view(B, C, -1).transpose(1, 2).contiguous()
    
    return final_feat_seq, canvas_mask

def test_apply_m1_scaling(image_path: str):
    """
    Visualization test for M1 Scaling logic.
    """
    if not os.path.exists(image_path):
        print(f"Error: M1 test image not found at {image_path}")
        return

    raw_img = Image.open(image_path).convert("RGB")
    configs = [
        (0.4, "black", "Shrink (0.4) / Black BG"),
        (0.7, "mean",  "Shrink (0.7) / Mean BG"),
        (1.0, "edge",  "Original (1.0) / Edge BG"),
        (1.5, "mean",  "Expand (1.5) / Mean BG")
    ]

    plt.figure(figsize=(20, 5))
    for i, (scale, mode, title) in enumerate(configs):
        processed_img_pil = apply_m1_scaling(raw_img, scale, 224, mode)
        img_array = np.array(processed_img_pil)

        plt.subplot(1, len(configs), i + 1)
        plt.imshow(img_array)
        plt.title(title)
        plt.axis('off')
        print(f"M1 Scenario '{title}': Output Size {processed_img_pil.size}")

    plt.suptitle(f"M1 Pixel Space Scaling - Source: {os.path.basename(image_path)}")
    plt.tight_layout()
    plt.savefig("m1_test_comparison.png")
    plt.show()

def test_apply_m2_latent_scaling(image_path: str):
    """
    Visualization test for M2 Latent Scaling logic.
    Simulates a feature map using RGB channels for visibility.
    """
    if not os.path.exists(image_path):
        print(f"Error: M2 test image not found at {image_path}")
        return

    # Simulate input features (B, N, C)
    # Using a 224x224 image resized to 16x16 grid to mimic DINOv2 latent space
    raw_img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_tensor = T.ToTensor()(raw_img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    
    # Simulate latent features [1, 3, 16, 16]
    feat_2d = F.interpolate(img_tensor, size=(16, 16), mode='bilinear') 
    B, C, H, W = feat_2d.shape
    object_feat = feat_2d.view(B, C, -1).transpose(1, 2)

    # Simulate object mask
    mask = torch.zeros((1, 1, 224, 224), device=DEVICE)
    mask[:, :, 40:184, 40:184] = 1.0

    scales = [0.5, 0.8, 1.2, 1.5]
    plt.figure(figsize=(20, 5))

    for i, scale in enumerate(scales):
        scaled_feat_seq, _ = apply_m2_latent_scaling(
            object_feat=object_feat,
            object_mask=mask,
            scale_factor=scale,
            target_latent_size=16
        )

        # Convert sequence back to 2D for visualization
        scaled_feat_2d = scaled_feat_seq.transpose(1, 2).view(1, 3, 16, 16)
        img_show = scaled_feat_2d.squeeze().permute(1, 2, 0).cpu().numpy()
        img_show = np.clip(img_show, 0, 1)

        plt.subplot(1, len(scales), i + 1)
        plt.imshow(img_show)
        plt.title(f"M2 Scale: {scale}")
        plt.axis('off')

    plt.suptitle("M2 Latent Space Scaling (Simulated 16x16 Features)")
    plt.tight_layout()
    plt.savefig("m2_test_comparison.png")
    plt.show()

def compute_mask_area_ratio(mask: torch.Tensor) -> float:
    """
    mask: (1, 1, H, W) or (1, H, W), values in {0,1}
    """
    if mask.dim() == 4:
        mask_2d = mask[0, 0]
    elif mask.dim() == 3:
        mask_2d = mask[0]
    else:
        raise ValueError("Unsupported mask shape")

    area = mask_2d.sum().item()
    total = mask_2d.numel()
    return float(area / max(total, 1))

def adaptive_target_size(
    mask: torch.Tensor,
    latent_hw: int,
    min_size: int = 10,
    max_ratio: float = 0.9,
    gamma: float = 0.5
) -> int:
    """
    gamma = 0.5 means sqrt(area_ratio)
    """
    area_ratio = compute_mask_area_ratio(mask)
    side_ratio = area_ratio ** gamma

    raw_size = latent_hw * side_ratio * max_ratio
    target_size = int(round(raw_size))

    return int(np.clip(target_size, min_size, latent_hw - 2))

if __name__ == "__main__":
    # Path configuration
    test_image_path = 'assets/group10/2.png'
    os.makedirs('debug', exist_ok=True)

    print(f"Running tests on device: {DEVICE}")

    # Run M1 scaling test
    print("\nExecuting M1 Scaling Test...")
    test_apply_m1_scaling(test_image_path)

    # Run M2 scaling test
    print("\nExecuting M2 Latent Scaling Test...")
    test_apply_m2_latent_scaling(test_image_path)