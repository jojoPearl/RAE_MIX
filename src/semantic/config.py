import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Always use float32 for geometry/masks/grids/coords
DTYPE_SAFE = torch.float32

# Use fp16 for model compute (autocast only)
DTYPE_MODEL = torch.float16

# Manual switch if you really want bf16
USE_BF16 = False
if USE_BF16 and torch.cuda.is_bf16_supported():
    DTYPE_MODEL = torch.bfloat16

# Backward compatible names (so you don't need to change all imports)
DTYPE = DTYPE_SAFE
DTYPE_MODERN = DTYPE_MODEL

H_PATCH = 32
W_PATCH = 32
