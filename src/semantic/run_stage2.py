import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda"
)
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import sys
import datetime
import os
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from torchvision.utils import save_image
from src.stage2.transport import create_transport, Sampler
from src.semantic.utils import cleanup_memory
from src.semantic.modelManager import ModelManager
from src.semantic.config import DEVICE, DTYPE, DTYPE_MODERN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = DEVICE
dtype = DTYPE

def run_diffusion_generation(base_fused_features, editing_strengths, total_steps, device, dtype):
    """
    Executes diffusion generation using a custom Euler loop that leverages the
    Transport class for drift calculation. This allows for precise control over 
    the starting time 't' while maintaining compatibility with different model 
    prediction types (velocity/noise/score).
    """
    manager = ModelManager(device=device)
    dit_model = manager.load_dit()
    if dit_model is None:
        raise RuntimeError("Failed to load DiT model.")

    # 1. Initialize Transport object to utilize its logic (e.g., drift calculation)
    # Ensure these parameters match your training configuration.
    transport = create_transport(
        path_type='Linear',       
        prediction='velocity',    
    )
    
    # 2. Get the drift function
    # This function abstracts the math: it converts model output (noise/score/velocity)
    # into the velocity vector required for the ODE.
    drift_fn = transport.get_drift()

    B, C, H, W = base_fused_features.shape
    dummy_y = torch.zeros(B, dtype=torch.long, device=device)
    model_kwargs = dict(y=dummy_y, s=None, mask=None)
    
    # Generate fixed initial noise for consistency across different strengths
    initial_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)
    sampled_latents_list = []

    print(f"--- Stage 2.1: Custom Transport Sampling ---")
    
    for start_t in editing_strengths:
        # Interpolate: Construct the starting latent at time t=start_t
        latents = (1.0 - start_t) * base_fused_features + start_t * initial_noise
        
        # Calculate proportional steps
        current_steps = int(total_steps * start_t)
        if current_steps < 1: 
            current_steps = 1
        
        # Create strict time grid from start_t down to 0.0
        timesteps = torch.linspace(start_t, 0.0, current_steps + 1, device=device, dtype=dtype)
        
        print(f"Strength t={start_t}: Running {current_steps} steps via Transport drift")

        with torch.no_grad():
            for i in range(current_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i+1]
                dt = t_next - t_curr 
                
                # Broadcast current time to batch size
                vec_t = torch.ones(B, device=device, dtype=dtype) * t_curr
                
                # Calculate velocity using the Transport logic
                # drift_fn handles the underlying math based on the model type
                velocity = drift_fn(latents, vec_t, dit_model, **model_kwargs)
                
                # Manual Euler Step: x_next = x_curr + v * dt
                latents = latents + velocity * dt
            
            # Post-loop processing
            final_latent = latents
            if final_latent.dim() == 5 and final_latent.shape[1] == 1:
                final_latent = final_latent.squeeze(1)
            
            sampled_latents_list.append(final_latent.cpu())

    # Cleanup
    del dit_model
    cleanup_memory()
    
    return sampled_latents_list

def run_decoding(sampled_latents_list, output_path, timestamp, device):
    """
    Step 2: Focus only on RAE Decoding.
    """
    manager = ModelManager(device=device)
    rae = manager.load_rae()
    if rae is None:
        raise RuntimeError("Failed to load RAE model.")

    generated_images = []
    print(f"--- Stage 2.2: Decoding Latents with RAE ---")

    for i, latents in enumerate(sampled_latents_list):
        latents = latents.to(device)
        with torch.no_grad():
            img = rae.decode(latents)
            img_cpu = img[0].cpu()
            generated_images.append(img_cpu)

        out_name = os.path.join(output_path, f"stage2_decoded_{i}_{timestamp}.png")
        save_image(img_cpu, out_name)
        print(f"Decoded image saved to: {out_name}")

        cleanup_memory()

    del rae
    cleanup_memory()
    return generated_images

def stage2_optimized(output_path, fused_path):
    device = DEVICE
    dtype = DTYPE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    editing_strengths = [0.1, 0.3, 0.4, 0.6, 0.8]
    num_steps = 50

    # 1. Load data
    data = torch.load(fused_path, map_location=device)
    base_fused_features = data['fused_features'].to(dtype=dtype)

    # 2. Part 1: Diffusion
    latents_list = run_diffusion_generation(
        base_fused_features, editing_strengths, num_steps, device, dtype
    )
    
    # 3. Clear base features to free memory
    del base_fused_features
    cleanup_memory()

    # 4. Part 2: Decoding
    images = run_decoding(latents_list, output_path, timestamp, device)

    # 5. Post-processing (Stitching)
    if images:
        stitched_image = torch.cat(images, dim=2)
        out_name = os.path.join(output_path, f"stitched_{timestamp}.png")
        save_image(stitched_image, out_name)
        print(f"Success. Saved to {out_name}")

if __name__ == "__main__":
    out_dir = "assets/group7/"
    fused_file = os.path.join(out_dir, "stage1_result","fused_results.pt")
    print(f"Processing fused file at: {fused_file}")
    if os.path.exists(fused_file):
        out_dir_stage2 = os.path.join(out_dir, "stage2_result")
        os.makedirs(out_dir_stage2, exist_ok=True)
        stage2_optimized(out_dir_stage2, fused_file)
    else:
        print(f"Fused file not found: {fused_file}. Please run Stage 1 first.")