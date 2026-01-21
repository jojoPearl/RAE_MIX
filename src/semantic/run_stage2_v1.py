import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda"
)
import torch
import sys
import datetime
import os
import glob
from torchvision.utils import save_image

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.stage2.transport import create_transport
from src.semantic.utils import cleanup_memory
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE

def run_diffusion_pass(file_paths, editing_strengths, total_steps, device, dtype):
    """
    Pass 1: Load DiT ONCE, process all files, return a list of latents.
    """
    manager = ModelManager(device=device)
    print("\n[Phase 1] Loading DiT Model...")
    dit_model = manager.load_dit()
    
    if dit_model is None:
        raise RuntimeError("Failed to load DiT model.")

    transport = create_transport(path_type='Linear', prediction='velocity')
    drift_fn = transport.get_drift()

    results_buffer = [] # Stores {'filename': str, 'latents_list': [Tensor]}

    print(f"[Phase 1] Starting Diffusion Batch for {len(file_paths)} files...")

    for f_idx, fused_path in enumerate(file_paths):
        print(f"  -> Processing file {f_idx+1}/{len(file_paths)}: {os.path.basename(fused_path)}")
        
        # Load the fused features
        data = torch.load(fused_path, map_location=device)
        base_fused_features = data['fused_features'].to(dtype=dtype)
        
        B, C, H, W = base_fused_features.shape
        dummy_y = torch.zeros(B, dtype=torch.long, device=device)
        model_kwargs = dict(y=dummy_y, s=None, mask=None)
        
        # Fixed initial noise
        initial_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)
        sampled_latents_list = []

        for start_t in editing_strengths:
            # Interpolate
            latents = (1.0 - start_t) * base_fused_features + start_t * initial_noise
            
            current_steps = int(total_steps * start_t)
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
                if final_latent.dim() == 5 and final_latent.shape[1] == 1:
                    final_latent = final_latent.squeeze(1)
                
                # Move to CPU to save VRAM for next iteration
                sampled_latents_list.append(final_latent.cpu())
        
        results_buffer.append({
            'file_path': fused_path,
            'latents_list': sampled_latents_list
        })
        
        # Small cleanup between files
        del base_fused_features, initial_noise, latents
        cleanup_memory()

    print("[Phase 1] Diffusion Pass Completed.")
    del dit_model
    cleanup_memory()
    
    return results_buffer


def run_decoding_pass(results_buffer, output_root_dir, device):
    """
    Pass 2: Load RAE ONCE, decode all buffered latents, save to folders.
    """
    manager = ModelManager(device=device)
    print("\n[Phase 2] Loading RAE Model...")
    rae = manager.load_rae()
    
    if rae is None:
        raise RuntimeError("Failed to load RAE model.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"[Phase 2] Decoding {len(results_buffer)} items...")

    for item in results_buffer:
        original_path = item['file_path']
        latents_list = item['latents_list']
        
        # Extract filename logic to create subfolder
        # e.g., "fused_results_scale_1.2.pt" -> folder "scale_1.2"
        filename = os.path.basename(original_path)
        base_name = os.path.splitext(filename)[0] # remove .pt
        
        # If the filename has "fused_results_", we can strip it for cleaner folder names
        folder_name = base_name.replace("fused_results_", "") 
        
        save_dir = os.path.join(output_root_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"  -> Decoding for: {folder_name}")
        
        decoded_imgs = []
        for i, latents in enumerate(latents_list):
            latents = latents.to(device) # Move back to GPU
            
            with torch.no_grad():
                img = rae.decode(latents)
                img_cpu = img[0].cpu()
                decoded_imgs.append(img_cpu)

            # Save individual image
            out_name = os.path.join(save_dir, f"result_strength_{i}_{timestamp}.png")
            save_image(img_cpu, out_name)
        
        # Save stitched preview
        if decoded_imgs:
            stitched = torch.cat(decoded_imgs, dim=2)
            stitch_name = os.path.join(save_dir, f"stitched_summary_{timestamp}.png")
            save_image(stitched, stitch_name)
            
        cleanup_memory()

    print("[Phase 2] Decoding Pass Completed.")
    del rae
    cleanup_memory()


def stage2_batch_process(stage1_output_dir, stage2_output_dir):
    # 1. Find all .pt files
    search_pattern = os.path.join(stage1_output_dir, "*.pt")
    fused_files = glob.glob(search_pattern)
    fused_files.sort() # Ensure consistent order

    if not fused_files:
        print(f"No .pt files found in {stage1_output_dir}")
        return

    print(f"Found {len(fused_files)} files to process: {[os.path.basename(f) for f in fused_files]}")

    # Config
    editing_strengths = [0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
    # editing_strengths = [0.9, 0.95]
    num_steps = 50

    # 2. Phase 1: Diffusion (Batch)
    results_buffer = run_diffusion_pass(
        fused_files, editing_strengths, num_steps, device, dtype
    )

    # 3. Phase 2: Decoding (Batch)
    run_decoding_pass(results_buffer, stage2_output_dir, device)
    
    print(f"\nAll Done! Results saved in: {stage2_output_dir}")


if __name__ == "__main__":
    # --- Configuration ---
    # The folder where stage1 saved the .pt files
    input_dir = "assets/group4/stage1_result1/"
    
    # The root folder where you want stage2 subfolders to be created
    output_root = "assets/group4/stage2_result1/"
    
    stage2_batch_process(input_dir, output_root)