import torch
import gc
import os
import sys
from contextlib import contextmanager

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from src.stage1.rae import RAE
from src.stage2.models.DDT import DiTwDDTHead

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
class ModelManager:
    def __init__(self, device: str, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.rae = None
        self.dit = None

    def load_rae(self) -> RAE:
        decoder_weights_path = 'models/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt'
        if not os.path.exists(decoder_weights_path):
            print(f"Error: Decoder weights file not found at {decoder_weights_path}")
            return None
        print(f"Loading RAE Model from {decoder_weights_path}...")
        rae_224 = RAE(
            encoder_cls='Dinov2withNorm',
            encoder_config_path='facebook/dinov2-with-registers-base',
            encoder_input_size=224,
            encoder_params={'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
            decoder_config_path='configs/decoder/ViTXL',
            pretrained_decoder_path='models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
            noise_tau=0.0,
            reshape_to_2d=True,
            normalization_stat_path='models/stats/dinov2/wReg_base/imagenet1k/stat.pt',
            eps=1e-5,
        ).to(self.device).eval()
        return rae_224.to(self.device).eval()

    def load_dit(self) -> DiTwDDTHead:
        model_params_224 = {
            'input_size': 16, 'patch_size': 1, 'in_channels': 768,
            'hidden_size': [1152, 2048], 'depth': [28, 2], 'num_heads': [16, 16],
            'mlp_ratio': 4.0, 'class_dropout_prob': 0.1, 'num_classes': 1000,
            'use_qknorm': False, 'use_swiglu': True, 'use_rope': True,
            'use_rmsnorm': True, 'wo_shift': False,'use_pos_embed': True
        }
        # model_params = {
        #     'input_size': 32, 'patch_size': 1, 'in_channels': 768,
        #     'hidden_size': [1152, 2048], 'depth': [28, 2], 'num_heads': [16, 16],
        #     'mlp_ratio': 4.0, 'class_dropout_prob': 0.1, 'num_classes': 1000,
        #     'use_qknorm': False, 'use_swiglu': True, 'use_rope': True,
        #     'use_rmsnorm': True, 'wo_shift': False, 'use_pos_embed': True
        # }
        print("Instantiating DiT model (DiTwDDTHead)...")
        with suppress_stdout():
            model = DiTwDDTHead(**model_params_224)
        try:
            # ckpt = torch.load('models/DiTs/Dinov2/wReg_base/ImageNet512/DiTDH-XL_ep400/stage2_model.pt',
            #                   map_location="cpu")
#             hf download nyu-visionx/RAE-collections \
#   DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt \
#   --local-dir models 
            ckpt = torch.load('models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt',
                              map_location="cpu")
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device).eval()
            print("DiT model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading DiT: {e}")
            return None