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
import scipy.ndimage as ndimage

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


def get_text_guided_coords(
    base_image,
    location_prompt: str,
    obj_h: int,
    obj_w: int,
    latent_hw: Tuple[int, int],
    avoid_mask: Optional[torch.Tensor] = None,   # [1,1,H,W] or [H,W] in [0,1]
    avoid_strength: float = 2.0,                 # higher => stronger repulsion
    avoid_thr: float = 0.2,                      # occupied threshold
):
    """
    Returns (ts_h, te_h, ts_w, te_w) in latent coordinates or None.
    """
    H, W = latent_hw

    # 1) Raw probability map in latent resolution
    heatmap = extract_semantic_mask_with_clipseg(base_image, location_prompt, (H, W), 0.0)
    heatmap = heatmap.float().cpu().numpy()

    # 2) Validity check
    if heatmap.max() < 0.05:
        return None

    # 3) Smooth heatmap
    heatmap_smoothed = ndimage.gaussian_filter(heatmap, sigma=1.0)

    # 4) Apply avoidance mask (repel from occupied regions)
    if avoid_mask is not None:
        if isinstance(avoid_mask, torch.Tensor):
            am = avoid_mask.detach().float().cpu()
            if am.ndim == 4:
                am = am[0, 0]
            am = am.numpy()
        else:
            am = np.asarray(avoid_mask, dtype=np.float32)

        if am.shape != (H, W):
            # If caller passed mismatched size, best effort resize with numpy/scipy
            # (Usually caller will pass correct latent size.)
            am = ndimage.zoom(am, (H / am.shape[0], W / am.shape[1]), order=1)

        occ = (am > avoid_thr).astype(np.float32)
        if occ.max() > 0:
            occ = ndimage.gaussian_filter(occ, sigma=1.5)
            # Penalize occupied regions. This keeps non-occupied peaks intact.
            heatmap_smoothed = heatmap_smoothed / (1.0 + avoid_strength * occ)

    # 5) Find peak
    idx = int(np.argmax(heatmap_smoothed))
    y, x = idx // W, idx % W

    ts_h = int(np.clip(y - obj_h // 2, 0, H - obj_h))
    ts_w = int(np.clip(x - obj_w // 2, 0, W - obj_w))

    print(f"[Smart Placement] Found target center at: ({y}, {x})")
    return ts_h, ts_h + obj_h, ts_w, ts_w + obj_w


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