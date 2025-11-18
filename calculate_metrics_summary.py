import json
import numpy as np
import sys
import os
from pathlib import Path

def calculate_summary(result_dir):
    result_path = Path(result_dir)
    metrics_file = result_path / 'metrics.json'
    fid_file = result_path / 'fid' / 'fid.txt'
    
    # Read metrics.json
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Extract values for each metric
    psnr_values = data['psnr']['mean']
    ssim_values = data['ssim']['mean'] 
    lpips_values = data.get('lpips', {}).get('mean', None)
    
    # Read FID score if exists
    fid_score = None
    if fid_file.exists():
        with open(fid_file, 'r') as f:
            fid_line = f.read().strip()
            fid_score = float(fid_line.split(':')[1].strip())
    else:
        print("⚠️  FID文件不存在，跳过FID指标显示")
    
    # Calculate overall statistics
    print("Results Summary:")
    print("=" * 40)
    print(f"PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"SSIM: {np.mean(ssim_values):.3f} ± {np.std(ssim_values):.3f}")
    if lpips_values is not None:
        print(f"LPIPS: {np.mean(lpips_values):.3f} ± {np.std(lpips_values):.3f}")
    else:
        print("LPIPS: 未计算 (使用 eval_fn_list=['psnr','ssim','lpips'] 来计算)")
    if fid_score is not None:
        print(f"FID: {fid_score:.4f}")
    else:
        print("FID: 未计算 (使用 eval_fid=true 来计算)")
    print(f"Number of images: {len(psnr_values)}")
    
    # Show per-image breakdown
    print("\nPer-image breakdown:")
    if lpips_values is not None:
        for i, (p, s, l) in enumerate(zip(psnr_values, ssim_values, lpips_values)):
            print(f"Image {i:2d}: PSNR={p:5.2f}, SSIM={s:.3f}, LPIPS={l:.3f}")
    else:
        for i, (p, s) in enumerate(zip(psnr_values, ssim_values)):
            print(f"Image {i:2d}: PSNR={p:5.2f}, SSIM={s:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_metrics_summary.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    calculate_summary(result_dir)