import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.semantic.config import DEVICE, DTYPE

_processor = None
_model = None


def _init_clipseg():
    global _processor, _model
    if _model is None or _processor is None:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        print(f"Initializing CLIPSeg model on {DEVICE}...")
        MODEL_ID = "CIDAS/clipseg-rd64-refined"
        _processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
        _model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
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
    ).to(DEVICE)

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

def extract_image_with_clipseg(image: Image.Image, target_text: str, threshold: float = 0.5) -> Image.Image:
    w, h = image.size
    
    mask_tensor = extract_semantic_mask_with_clipseg(image, target_text, (h, w), threshold)
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    img_np = np.array(image.convert("RGB"))
    rgba_np = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_np[..., :3] = img_np
    rgba_np[..., 3] = mask_np
    
    return Image.fromarray(rgba_np)

import scipy.ndimage as ndimage

def get_text_guided_coords(base_image, location_prompt, obj_h, obj_w, latent_hw):
    H, W = latent_hw
    # 1. Get raw probability map
    heatmap = extract_semantic_mask_with_clipseg(base_image, location_prompt, (H, W), 0.0)
    heatmap = heatmap.float().cpu().numpy()
    
    # 2. Check validity
    if heatmap.max() < 0.05:
        return None
    
    # 3. Key improvement: use Gaussian blur to smooth heatmap and eliminate point noise
    heatmap_smoothed = ndimage.gaussian_filter(heatmap, sigma=1.0)
    
    # 4. Find the strongest response center point
    idx = np.argmax(heatmap_smoothed)
    y, x = idx // W, idx % W
    
    ts_h = np.clip(y - obj_h // 2, 0, H - obj_h)
    ts_w = np.clip(x - obj_w // 2, 0, W - obj_w)
    
    print(f"[Smart Placement] Found target center at: ({y}, {x})")
    
    return int(ts_h), int(ts_h + obj_h), int(ts_w), int(ts_w + obj_w)

# --- Usage example ---
if __name__ == "__main__":
    # Load local image
    test_img_path = "/home/bjia-25/workspace/papers/RAE/code/rae_project/RAE/assets/group9/2.png"
    input_image = Image.open(test_img_path).convert("RGB")
    
    # Extract object using text prompt
    result_image = extract_image_with_clipseg(input_image, target_text="animal", threshold=0.2)
    
    # Save and display result
    result_image.save("clipseg_result.png")
    result_image.show()
    print("Extraction complete. Result saved as 'clipseg_result.png'")