import json
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from piq import LPIPS
from tqdm import tqdm

def load_image_as_tensor(image_path):
    """Load image and convert to tensor [-1, 1]"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0  # [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
    img = img * 2.0 - 1.0  # [-1, 1]
    return img.unsqueeze(0)  # [1, C, H, W]

def norm_for_lpips(x):
    """Normalize tensor from [-1, 1] to [0, 1] for LPIPS calculation"""
    return (x * 0.5 + 0.5).clip(0, 1)

def calculate_lpips_for_results(result_dir, use_gpu=True, force_calculate=False):
    """Calculate LPIPS for existing results and update metrics.json"""
    result_path = Path(result_dir)
    metrics_file = result_path / 'metrics.json'
    
    if not metrics_file.exists():
        print(f"❌ metrics.json not found in {result_dir}")
        return
    
    # Load existing metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if 'lpips' in data and not force_calculate:
        print("✅ LPIPS already exists in metrics.json")
        print("💡 Use --force-calculate to recalculate")
        return
    elif 'lpips' in data and force_calculate:
        print("🔄 LPIPS exists but force recalculating...")
    
    # Initialize LPIPS
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    lpips_fn = LPIPS(replace_pooling=True, reduction='none').to(device)
    print(f"🔧 Using device: {device}")
    
    # Find GT and sample images
    samples_dir = result_path / 'samples'
    if not samples_dir.exists():
        print(f"❌ samples directory not found in {result_dir}")
        return
    
    # Get image files and determine structure
    sample_files = sorted([f for f in samples_dir.glob('*.png')])
    if not sample_files:
        print(f"❌ No PNG files found in {samples_dir}")
        return
    
    # Parse file structure to get GT images
    # Assume GT images are in the grid_results.png or we need to load from dataset
    grid_file = result_path / 'grid_results.png'
    if not grid_file.exists():
        print(f"❌ grid_results.png not found, cannot extract GT images")
        return
    
    # Load config to get dataset info
    config_file = result_path / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get dataset
        from data import get_dataset
        dataset = get_dataset(**config['data'])
        total_number = len(dataset)
        gt_images = dataset.get_data(total_number, 0)  # [B, C, H, W] tensor
    else:
        print(f"❌ config.yaml not found, cannot load GT images")
        return
    
    # Group sample files by image index
    sample_groups = {}
    for f in sample_files:
        # Parse filename: 00000_run0000.png
        parts = f.stem.split('_')
        img_idx = int(parts[0])
        run_idx = int(parts[1].replace('run', ''))
        
        if img_idx not in sample_groups:
            sample_groups[img_idx] = {}
        sample_groups[img_idx][run_idx] = f
    
    # Calculate LPIPS for each image and run
    lpips_results = []
    
    print(f"📊 Calculating LPIPS for {len(sample_groups)} images...")
    
    with tqdm(total=len(sample_groups), desc="LPIPS", unit="img") as pbar:
        for img_idx in sorted(sample_groups.keys()):
            gt_tensor = gt_images[img_idx:img_idx+1].to(device)  # [1, C, H, W]
            
            img_lpips = []
            for run_idx in sorted(sample_groups[img_idx].keys()):
                sample_file = sample_groups[img_idx][run_idx]
                sample_tensor = load_image_as_tensor(sample_file).to(device)
                
                # Calculate LPIPS (normalize to [0, 1] like eval.py)
                with torch.no_grad():
                    gt_norm = norm_for_lpips(gt_tensor)
                    sample_norm = norm_for_lpips(sample_tensor)
                    lpips_score = lpips_fn(gt_norm, sample_norm)
                    img_lpips.append(lpips_score.item())
            
            lpips_results.append(img_lpips)
            pbar.update(1)
    
    # Convert to same format as other metrics
    lpips_results = np.array(lpips_results)  # [num_images, num_runs]
    
    # Add LPIPS to metrics data
    data['lpips'] = {
        'sample': lpips_results.tolist(),
        'mean': lpips_results.mean(axis=1).tolist(),
        'std': lpips_results.std(axis=1).tolist(),
        'max': lpips_results.max(axis=1).tolist(),
        'min': lpips_results.min(axis=1).tolist()
    }
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ LPIPS calculated and saved to {metrics_file}")
    print(f"📈 Average LPIPS: {np.mean(lpips_results):.4f} ± {np.std(lpips_results):.4f}")
    
    # Print per-image LPIPS results
    print("\n📋 Per-image LPIPS results:")
    lpips_means = lpips_results.mean(axis=1)
    for img_idx, lpips_mean in enumerate(lpips_means):
        runs_str = ", ".join([f"{val:.4f}" for val in lpips_results[img_idx]])
        print(f"Image {img_idx:2d}: mean={lpips_mean:.4f} [runs: {runs_str}]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_lpips.py <result_directory> [--cpu] [--force-calculate]")
        print("Example: python calculate_lpips.py results/pixel/ffhq/251118-140325/phase_retrieval")
        print("Options:")
        print("  --cpu              Use CPU instead of GPU")
        print("  --force-calculate  Recalculate even if LPIPS already exists")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    use_gpu = '--cpu' not in sys.argv
    force_calculate = '--force-calculate' in sys.argv
    
    calculate_lpips_for_results(result_dir, use_gpu, force_calculate)