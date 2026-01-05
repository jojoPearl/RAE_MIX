import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import sys
from typing import Tuple, Dict, Callable, Optional, Union
from math import sqrt
import numpy as np
import datetime
import time
from pathlib import Path
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
from src.semantic.utils import cleanup_memory, load_and_transform, _calculate_target_coords, _calculate_dynamic_coords
from src.semantic.resize import apply_m1_scaling
from src.semantic.ModelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE

H_PATCH = 32
W_PATCH = 32

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


def semantic_placement_fusion(
        base_feat: torch.Tensor,
        object_feat: torch.Tensor,
        object_mask: torch.Tensor,
        target_size: int = 16,
        target_area: str = 'bottom_right',
        blend_strength: float = 1.0,
        use_color_matching: bool = False,
        debug: bool = False  # 添加调试模式
) -> Tuple[torch.Tensor, int, int]:
    """
    V5 硬编码版 (Hard-Coded):
    完全放弃统计匹配，直接像贴纸一样把物体特征“硬”覆盖到背景上。
    用于调试特征是否本身有问题。
    """
    B, N, C = base_feat.shape
    H = W = int(math.sqrt(N))

    if debug:
        print(f"[Debug] Base feat shape: {base_feat.shape}")
        print(f"[Debug] Object feat shape: {object_feat.shape}")
        print(f"[Debug] Object mask shape: {object_mask.shape}")
        print(f"[Debug] Target area: {target_area}, Target size: {target_size}")

    base_2d = base_feat.transpose(1, 2).view(B, C, H, W).clone()
    object_2d = object_feat.transpose(1, 2).view(B, C, H, W)

    # 1. Mask 处理 (保持最基础的缩放)
    if object_mask.shape[1] > 1:
        object_mask = object_mask[:, 0:1, :, :]
    
    # 获取全尺寸 Mask
    mask_2d = F.interpolate(object_mask, size=(H, W), mode='bilinear', align_corners=False)

    # 2. 提取物体特征
    clean_object_2d = object_2d * mask_2d

    # 3. 强制缩放到目标尺寸
    target_object_feat = F.interpolate(
        clean_object_2d,
        size=(target_size, target_size),
        mode='nearest'
    )

    target_mask = F.interpolate(
        mask_2d,
        size=(target_size, target_size),
        mode='nearest'
    )

    # 4. 生成硬 Mask (非黑即白)
    binary_mask = (target_mask > 0.1).float()

    if debug:
        print(f"[Debug] Binary mask area: {binary_mask.mean().item():.4f}")

    # 5. 计算坐标
    ts_h, te_h, ts_w, te_w = _calculate_target_coords(H, W, target_size, target_area)
    
    # 边界检查
    actual_h = te_h - ts_h
    actual_w = te_w - ts_w
    
    if debug:
        print(f"[Debug] Target coords: H[{ts_h}:{te_h}], W[{ts_w}:{te_w}]")
        print(f"[Debug] Actual region size: {actual_h}x{actual_w}")

    # 检查目标区域是否有效
    if ts_h >= te_h or ts_w >= te_w or ts_h < 0 or ts_w < 0 or te_h > H or te_w > W:
        print(f"[Error] Invalid target region: H[{ts_h}:{te_h}], W[{ts_w}:{te_w}] for HxW={H}x{W}")
        return base_feat, H, W

    # 6. 提取背景
    base_target = base_2d[:, :, ts_h:te_h, ts_w:te_w].clone()

    # 确保尺寸匹配
    if target_object_feat.shape[2:] != base_target.shape[2:]:
        print(f"[Warning] Size mismatch: object {target_object_feat.shape[2:]}, base {base_target.shape[2:]}")
        # 调整对象特征大小以匹配目标区域
        target_object_feat = F.interpolate(
            target_object_feat,
            size=base_target.shape[2:],
            mode='nearest'
        )
        binary_mask = F.interpolate(
            binary_mask,
            size=base_target.shape[2:],
            mode='nearest'
        )

    # 7. 硬编码核心：直接替换
    boost_factor = 1.2  # 稍微增强特征
    final_object = target_object_feat * boost_factor

    # 硬融合：二进制混合
    hard_fused_patch = binary_mask * final_object + (1 - binary_mask) * base_target

    if debug:
        print(f"[Debug] Fusion patch shape: {hard_fused_patch.shape}")

    # 8. 写回
    fused_2d = base_2d.clone()
    fused_2d[:, :, ts_h:te_h, ts_w:te_w] = hard_fused_patch

    fused_seq = fused_2d.view(B, C, N).transpose(1, 2).contiguous()
    return fused_seq, H, W

def hard_semantic_fusion(
    base_latent: torch.Tensor, 
    object_latent: torch.Tensor, 
    object_mask: torch.Tensor, 
    target_size: int = 16, 
    target_area: str = 'bottom_right'
) -> torch.Tensor:
    B, C, H, W = base_latent.shape
    
    # 1. 找到物体在 Mask 中的有效包围框 (裁剪掉多余背景)
    # 这一步是为了只提取“狗/猫”本身，不带入原图的背景特征
    mask_2d = F.interpolate(object_mask, size=(H, W), mode='bilinear')
    mask_np = mask_2d.squeeze().cpu().numpy()
    rows, cols = np.where(mask_np > 0.1)
    
    if len(rows) == 0: 
        print("[Warning] Mask is empty in Latent Space.")
        return base_latent

    rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
    
    # 2. 提取有效区域特征 (保持比例)
    obj_feat_crop = object_latent[:, :, rmin:rmax+1, cmin:cmax+1]
    obj_mask_crop = mask_2d[:, :, rmin:rmax+1, cmin:cmax+1]

    # 3. 计算缩放比例：将长边限制在 target_size 内
    h_orig, w_orig = obj_feat_crop.shape[2], obj_feat_crop.shape[3]
    scale = target_size / max(h_orig, w_orig)
    h_new, w_new = max(1, int(h_orig * scale)), max(1, int(w_orig * scale))

    # 4. 执行不失真的重采样 (M2 Scaling 核心)
    target_object_feat = F.interpolate(obj_feat_crop, size=(h_new, w_new), mode='bicubic')
    target_mask = F.interpolate(obj_mask_crop, size=(h_new, w_new), mode='nearest')

    # 5. 计算动态放置坐标
    ts_h, te_h, ts_w, te_w = _calculate_dynamic_coords(H, W, h_new, w_new, target_area)

    # 6. 硬对齐覆盖
    fused_latent = base_latent.clone()
    # 截取背景中同样大小的坑
    base_patch = fused_latent[:, :, ts_h:te_h, ts_w:te_w]
    
    # 容错：确保插值后尺寸完全一致
    target_object_feat = F.interpolate(target_object_feat, size=base_patch.shape[2:])
    target_mask = F.interpolate(target_mask, size=base_patch.shape[2:])

    # 混合：物体 + 坑里的原始背景
    # 这样做可以确保物体边缘外的背景不会被干扰，减少“虚影”
    fused_patch = target_mask * target_object_feat + (1 - target_mask) * base_patch
    fused_latent[:, :, ts_h:te_h, ts_w:te_w] = fused_patch
    
    return fused_latent

def stage1_extract_features(rae, base_img_tensor, replace_pil_image, target_area, target_text, output_path):
    """
    Stage 1 改进版：结合 CLIPSeg 精确提取与 Latent 硬融合
    """
    print(f"[Stage 1] Using CLIPSeg to extract '{target_text}'...")
    
    # 1. 图像层面：精确抠图 (使用第一版的逻辑)
    # scale_factor 控制物体在生成的 224x224 框里的大小
    object_tensor, object_mask = get_cropped_object_tensor(
        raw_image=replace_pil_image, 
        target_text=target_text,
        scale_factor=1.2  # 这里的缩放会直接影响 Latent 里的物体比例
    )

    # 调试保存
    debug_dir = os.path.join(output_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    save_image(object_tensor, os.path.join(debug_dir, f"crop_obj_{timestamp}.png"))

    with torch.no_grad():
        # 2. 编码：将图像转为 Latent Space 向量
        base_latent = rae.encode(base_img_tensor)
        object_latent = rae.encode(object_tensor)
        
        B, C, H, W = base_latent.shape

        print(f"[Stage 1] Fusing latents at {target_area} with size 8x8...")
        fused_latent_2d = hard_semantic_fusion(
            base_latent=base_latent,
            object_latent=object_latent,
            object_mask=object_mask,
            target_size=16, # 占多少个 patch
            target_area=target_area
        )

        # 4. 转换回序列格式 (Transformer 需要 B, N, C)
        fused_features_seq = fused_latent_2d.flatten(2).transpose(1, 2).contiguous()

        # 5. 预览检查
        check_img = rae.decode(fused_latent_2d)
        save_image(check_img, os.path.join(debug_dir, f"latent_fusion_check_{timestamp}.png"))
        print(f"Fusion preview saved to debug directory.")

    return fused_features_seq, fused_latent_2d

def stage1_extract_features_ori(rae, base_img_tensor, replace_pil_image, target_area, target_text, output_path):
    """Stage 1: 提取和融合特征 - 修复版本"""
    print(f"Preprocessing: Cropping '{target_text}' from image...")
    
    # 添加缩放因子参数
    object_tensor, object_mask = get_cropped_object_tensor(
        raw_image=replace_pil_image, 
        target_text=target_text,
        scale_factor=0.7  # 调整物体大小
    )

    # Save intermediate images for debugging
    debug_dir = os.path.join(output_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    save_image(object_tensor, os.path.join(debug_dir, f"debug_object_{timestamp}.png"))
    save_image(object_mask, os.path.join(debug_dir, f"debug_mask_{timestamp}.png"))

    with torch.no_grad():
        # Encode base image
        base_features_seq = rae.encode(base_img_tensor).contiguous().view(1, 768, -1).transpose(1, 2)

        # Encode object image
        object_features_seq = rae.encode(object_tensor).contiguous().view(1, 768, -1).transpose(1, 2)

        B, N, C = base_features_seq.shape

        # 解码对象特征用于检查
        object_features_2d = object_features_seq.transpose(1, 2).view(B, C, 32, 32)
        obj_check = rae.decode(object_features_2d)
        save_image(obj_check, os.path.join(debug_dir, f"encoded_object_{timestamp}.png"))

    print(f"Running CLIPSeg Semantic Placement Fusion: '{target_text}' to base {target_area}")

    # 修复：只尝试一次，并确保返回正确的结果
    blend_strength = 0.2
    print(f"\nUsing blend strength: {blend_strength}")

    fused_features_seq, H_fused, W_fused = semantic_placement_fusion(
        base_feat=base_features_seq,
        object_feat=object_features_seq,
        object_mask=object_mask,
        target_size=16,
        target_area=target_area,
        blend_strength=blend_strength,
        debug=True  # 开启调试模式
    )

    fused_features_2d = fused_features_seq.transpose(1, 2).view(B, C, H_fused, W_fused)

    # 检查融合结果
    with torch.no_grad():
        check_img = rae.decode(fused_features_2d)
        save_image(check_img, os.path.join(debug_dir, f"final_fusion_{timestamp}.png"))
        print(f"Saved fusion check to debug directory")

    return fused_features_seq, fused_features_2d


def create_mock_transport_and_sampler(device, num_steps, time_dist_shift):
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
    base_fused_features_2d = data['fused_features'].to(dtype=dtype)
    B, C, H, W = base_fused_features_2d.shape

    # editing_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
    editing_strengths = [0.8]
    generated_images_list = []
    num_steps = 15  # Slightly more steps for better quality
    sample_fn = create_mock_transport_and_sampler(device, num_steps, time_dist_shift=1.0)

    manager = ModelManager(device=device)
    dit_model = manager.load_dit()

    if dit_model is None:
        print("Failed to load DiT model - aborting generation.")
        return
    else:
        print("DiT model loaded successfully for Stage 2.")

    model_fwd = dit_model.forward
    dummy_y = torch.zeros(B, dtype=torch.long, device=base_fused_features_2d.device)
    model_kwargs = dict(y=dummy_y, s=None, mask=None)

    rae = manager.load_rae()

    print(f"Running Stage 2 with {len(editing_strengths)} Editing Strengths (Steps: {num_steps})")
    initial_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)

    for t in editing_strengths:
        # Smarter blending: use different noise schedules based on editing strength
        if t < 0.3:
            # Light editing: more preservation of original features
            noise_scale = 0.3
        elif t < 0.7:
            # Moderate editing: balanced
            noise_scale = 0.5
        else:
            # Strong editing: more generation
            noise_scale = 0.8

        initial_latents = (1.0 - noise_scale * t) * base_fused_features_2d + noise_scale * t * initial_noise

        print(f"Generating for editing_strength t = {t:.1f} (noise_scale={noise_scale})")

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

        print(f"Saved: {t_filename}")

        cleanup_memory()

    del base_fused_features_2d
    cleanup_memory()

    if generated_images_list:
        stitched_image = torch.cat(generated_images_list, dim=2)
        output_filename = f"{output_path}stitched_output_{timestamp}.png"
        save_image(stitched_image, output_filename)

        print(f"Final Output")
        print(f"Successfully stitched and saved all {len(editing_strengths)} levels.")
        print(f"Saved to: {output_filename}")
    else:
        print("Error: No images were generated to stitch.")

    del dit_model, rae
    manager.cleanup()
    cleanup_memory()


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
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "debug"), exist_ok=True)

    # Run stage1 only first to check the fusion
    fused_path = os.path.join(output_path, "fused_results.pt")
    # stage1(base_image_path, replace_image_path, output_path, target_area, target_text, fused_path)

    cleanup_memory()

    # Uncomment to run stage2
    stage2(output_path, fused_path)