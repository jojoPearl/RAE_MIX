import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Tuple, Dict, Callable, Optional
from math import sqrt
import numpy as np
import gc
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.semantic.config import DEVICE, DTYPE

def cleanup_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def load_and_transform(path: str, target_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0)
    tensor = tensor.to(device=DEVICE, dtype=DTYPE)
    return tensor

def _calculate_dynamic_coords(H: int, W: int, h_new: int, w_new: int, area: str) -> Tuple[int, int, int, int]:
    """
    Calculate placement coordinates on the background (H, W) based on the dynamic object size (h_new, w_new).
    """
    # Ensure object size does not exceed background size
    h_new = min(h_new, H)
    w_new = min(w_new, W)

    if area == 'top_left':
        ts_h, ts_w = 0, 0
    elif area == 'top_right':
        ts_h, ts_w = 0, W - w_new
    elif area == 'bottom_left':
        ts_h, ts_w = H - h_new, 0
    elif area == 'center':
        ts_h, ts_w = (H - h_new) // 2, (W - w_new) // 2
    elif area == 'bottom_right':
        ts_h, ts_w = H - h_new, W - w_new
    else:
        # Default to bottom-right
        ts_h, ts_w = H - h_new, W - w_new

    te_h, te_w = ts_h + h_new, ts_w + w_new
    
    # Final boundary check to prevent slice overflow
    te_h = min(te_h, H)
    te_w = min(te_w, W)
    
    return ts_h, te_h, ts_w, te_w