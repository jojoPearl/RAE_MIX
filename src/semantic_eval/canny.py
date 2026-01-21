import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import sys
import datetime
import os
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# --- Project Imports ---
from src.stage2.transport import create_transport
from src.stage1.rae import RAE
from src.semantic.utils import cleanup_memory, load_and_transform
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE, DTYPE_MODERN

# --- Config ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE
t_size = 448 # Image target size

def run_integrated_pipeline(
    base_image_path, 
    output_dir, 
    editing_strengths=[0.2, 0.4, 0.6], 
    num_steps=50
):
    """
    Runs the full pipeline in 3 memory-efficient phases:
    1. RAE Encode: Image -> Latent (Save Preview) -> Unload RAE
    2. DiT Diffusion: Latent -> Diffused Latents -> Unload DiT
    3. RAE Decode: Diffused Latents -> Final Images
    """
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting Pipeline for: {os.path.basename(base_image_path)} ===")

    # ==========================================
    # Phase 1: RAE Encoding & Preview
    # ==========================================
    print("\n[Phase 1] Loading RAE for Encoding...")
    manager = ModelManager(device=device)
    rae = manager.load_rae()
    cleanup_memory()
    
    # Load Image
    try:
        base_img_tensor = load_and_transform(base_image_path, t_size)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Encode
    with torch.amp.autocast('cuda', dtype=DTYPE_MODERN):
        base_latent = rae.encode(base_img_tensor) # Shape: [B, C, H, W]
        
        # Save Stage 1 Reconstruction Preview
        print("  -> Generating Stage 1 Preview (Reconstruction)...")
        reconstruction = rae.decode(base_latent)
        save_image(
            reconstruction.float(), 
            os.path.join(output_dir, f"stage1_preview_{timestamp}.png")
        )

    # Move latent to CPU temporarily to clear VRAM fully
    base_latent_cpu = base_latent.cpu()
    
    print("[Phase 1] Done. Unloading RAE...")
    del rae, reconstruction, base_img_tensor, base_latent
    cleanup_memory()

    # ==========================================
    # Phase 2: DiT Diffusion Process
    # ==========================================
    print("\n[Phase 2] Loading DiT for Diffusion...")
    dit_model = manager.load_dit()
    if dit_model is None: raise RuntimeError("Failed to load DiT")
    
    transport = create_transport(path_type='Linear', prediction='velocity')
    drift_fn = transport.get_drift()
    
    # Bring latent back to GPU
    base_latent = base_latent_cpu.to(device, dtype=dtype)
    B, C, H, W = base_latent.shape
    
    processed_latents = [] # Store results here (on CPU)

    print(f"  -> Processing strengths: {editing_strengths}")
    
    # Prepare DiT args
    dummy_y = torch.zeros(B, dtype=torch.long, device=device)
    model_kwargs = dict(y=dummy_y, s=None, mask=None)
    
    # Fixed initial noise for consistency across strengths
    initial_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)

    for start_t in editing_strengths:
        print(f"    -> Running Diffusion (Strength: {start_t})...")
        
        # Mix Noise
        latents = (1.0 - start_t) * base_latent + start_t * initial_noise
        
        current_steps = int(num_steps * start_t)
        if current_steps < 1: current_steps = 1
        
        timesteps = torch.linspace(start_t, 0.0, current_steps + 1, device=device, dtype=dtype)
        
        with torch.no_grad():
            for i in range(current_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i+1]
                dt = t_next - t_curr 
                vec_t = torch.ones(B, device=device, dtype=dtype) * t_curr
                
                velocity = drift_fn(latents, vec_t, dit_model, **model_kwargs)
                latents = latents + velocity * dt
            
            final_latent = latents
            # Handle potential dimension squeeze if needed (DiT output sometimes varies)
            if final_latent.dim() == 5 and final_latent.shape[1] == 1:
                final_latent = final_latent.squeeze(1)

            # Store on CPU
            processed_latents.append({
                'strength': start_t,
                'latent': final_latent.cpu()
            })

    print("[Phase 2] Done. Unloading DiT...")
    del dit_model, base_latent, initial_noise, latents
    cleanup_memory()

    # ==========================================
    # Phase 3: RAE Decoding Final Results
    # ==========================================
    print("\n[Phase 3] Reloading RAE for Final Decoding...")
    rae = manager.load_rae()
    cleanup_memory()
    
    stitched_images = []

    for item in processed_latents:
        strength = item['strength']
        latent = item['latent'].to(device, dtype=dtype)
        
        with torch.no_grad():
            # Decode
            img = rae.decode(latent)
            img_cpu = img[0].cpu()
            
            # Save Individual Image
            filename = f"final_strength_{strength}_{timestamp}.png"
            save_path = os.path.join(output_dir, filename)
            save_image(img_cpu, save_path)
            print(f"  -> Saved: {filename}")
            
            stitched_images.append(img_cpu)

    # Optional: Save a stitched summary of all strengths
    if stitched_images:
        print("  -> Saving Summary Strip...")
        stitched = torch.cat(stitched_images, dim=2) # Concat horizontally
        save_image(stitched, os.path.join(output_dir, f"summary_strip_{timestamp}.png"))

    print("\n=== Pipeline Completed Successfully ===")
    del rae
    cleanup_memory()


if __name__ == "__main__":
    # --- Settings ---
    base_image_path = "assets/out_ctrl/r_depth.png"
    output_path = "assets/out_ctrl/base_latent_output/"
    
    # 0.0 = No change, 1.0 = Full noise/Hallucination
    # strengths = [0.2, 0.4, 0.6, 0.8, 0.9] 
    strengths = [0.9, 0.95] 
    
    if not os.path.exists(base_image_path):
        print(f"Error: Base image not found at {base_image_path}")
    else:
        run_integrated_pipeline(
            base_image_path, 
            output_path, 
            editing_strengths=strengths,
            num_steps=50
        )