# src/config.py
import torch

# Centralized device and dtype configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DTYPE_MODERN = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


H_PATCH = 32
W_PATCH = 32