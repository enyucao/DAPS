import os
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
from tqdm import tqdm

def evaluate_daps_results(results_dir="./results/gaussian_deblur_quick"):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # DAPS saves results in different structure
    gt_dir = Path(results_dir) / "gt"
    recon_dir = Path(results_dir) / "recon"
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    
    print(f"Evaluating {len(image_files)} images from {results_dir}...")
    print(f"DAPS parameters: 5 diffusion steps + 50 annealing steps")
    print(f"Task: Gaussian blur with kernel_size=61, intensity=3.0, sigma=0.05")
    
    for fname in tqdm(image_files):
        # Load images
        gt = plt.imread(gt_dir / fname)[:, :, :3]
        recon = plt.imread(recon_dir / fname)[:, :, :3]
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(gt, recon)
        psnr_list.append(psnr)
        
        # Calculate SSIM
        ssim = structural_similarity(gt, recon, multichannel=True, channel_axis=2, data_range=1.0)
        ssim_list.append(ssim)
        
        # Calculate LPIPS
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).to(device)
        recon_tensor = torch.from_numpy(recon).permute(2, 0, 1).to(device)
        
        gt_tensor = gt_tensor.view(1, 3, 256, 256) * 2. - 1.
        recon_tensor = recon_tensor.view(1, 3, 256, 256) * 2. - 1.
        
        lpips_score = loss_fn_vgg(recon_tensor, gt_tensor)
        lpips_list.append(lpips_score.item())
    
    # Calculate averages
    psnr_avg = np.mean(psnr_list)
    ssim_avg = np.mean(ssim_list)
    lpips_avg = np.mean(lpips_list)
    
    print(f"\nDAPS Results:")
    print(f"PSNR: {psnr_avg:.4f} ± {np.std(psnr_list):.4f}")
    print(f"SSIM: {ssim_avg:.4f} ± {np.std(ssim_list):.4f}")
    print(f"LPIPS: {lpips_avg:.4f} ± {np.std(lpips_list):.4f}")
    
    return {
        'psnr': psnr_avg,
        'ssim': ssim_avg,
        'lpips': lpips_avg,
        'psnr_std': np.std(psnr_list),
        'ssim_std': np.std(ssim_list),
        'lpips_std': np.std(lpips_list)
    }

if __name__ == "__main__":
    evaluate_daps_results()