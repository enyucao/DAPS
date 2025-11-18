import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from data import get_dataset
from evaluate_fid import calculate_fid
from eval import get_eval_fn_cmp, Evaluator
import yaml
import argparse

def tensor_to_pils(x):
    """[B, C, H, W] tensor -> list of pil images"""
    pils = []
    for x_ in x:
        np_x = ((x_ * 0.5 + 0.5).clip(0, 1)).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils

def calculate_fid_for_result(result_dir, data=None, start_id=None, end_id=None, batch_size=100):
    result_path = Path(result_dir)
    metrics_file = result_path / 'metrics.json'
    samples_dir = result_path / 'samples'
    fid_dir = result_path / 'fid'
    
    # Read metrics to get best samples
    with open(metrics_file, 'r') as f:
        results = json.load(f)
    
    # Load all sample images
    sample_files = sorted(list(samples_dir.glob('*.png')))
    
    # Group by image index and run
    samples_by_image = {}
    for sample_file in sample_files:
        parts = sample_file.stem.split('_')
        img_idx = int(parts[0])
        run_idx = int(parts[1][3:])  # remove 'run' prefix
        
        if img_idx not in samples_by_image:
            samples_by_image[img_idx] = {}
        samples_by_image[img_idx][run_idx] = sample_file
    
    # Select best samples based on main evaluation metric (like posterior_sample.py)
    # Find the main evaluation metric
    main_eval_fn_name = list(results.keys())[0]  # First metric is main
    eval_values = np.array(results[main_eval_fn_name]['sample'])  # [B, num_runs]
    
    # Determine if higher or lower is better
    if main_eval_fn_name in ['psnr', 'ssim']:
        best_idx = np.argmax(eval_values, axis=1)  # Higher is better
    else:  # lpips
        best_idx = np.argmin(eval_values, axis=1)  # Lower is better
    
    print(f'Using {main_eval_fn_name} as selection criterion')
    
    # Create FID directory and best_sample subdirectory
    fid_dir.mkdir(exist_ok=True)
    best_sample_dir = fid_dir / 'best_sample'
    best_sample_dir.mkdir(exist_ok=True)
    
    # Copy best samples
    copied_count = 0
    for img_idx in range(len(best_idx)):
        best_run = best_idx[img_idx]
        if img_idx in samples_by_image and best_run in samples_by_image[img_idx]:
            src_file = samples_by_image[img_idx][best_run]
            dst_file = best_sample_dir / f'{img_idx:05d}.png'
            
            # Copy file
            img = Image.open(src_file)
            img.save(dst_file)
            copied_count += 1
    
    print(f'Copied {copied_count} best samples to {best_sample_dir}')
    
    # Calculate FID
    print('Calculating FID...')
    
    # Load config to get real dataset path
    config_file = result_path / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        real_data_config = config['data']
        # Use command line args if provided, otherwise use config
        if data:
            # If data name is provided via command line, construct path
            real_root = f'dataset/{data}'
        else:
            # Use the exact root from config
            real_root = real_data_config.get('root', 'dataset/test-ffhq')
        real_resolution = real_data_config.get('resolution', 256)
        real_start_id = start_id if start_id is not None else real_data_config.get('start_id', None)
        real_end_id = end_id if end_id is not None else real_data_config.get('end_id', None)
        print(f'Using resolution from config: {real_resolution}')
    else:
        print('❌ config.yaml not found, using provided or default values')
        if data:
            real_root = f'dataset/{data}'
        else:
            real_root = 'dataset/test-ffhq'
        real_resolution = 256
        real_start_id = start_id
        real_end_id = end_id
        print(f'Using default resolution: {real_resolution}')
    
    print(f'Real dataset path: {real_root}')
    print(f'Checking if path exists: {Path(real_root).exists()}')
    
    # Get datasets exactly like posterior_sample.py
    fake_dataset = get_dataset(real_data_config['name'], resolution=real_resolution, root=str(best_sample_dir))
    
    # Use the SAME dataset configuration as the original experiment
    real_dataset = get_dataset(**real_data_config)
    
    print(f'Fake dataset size: {len(fake_dataset)}')
    print(f'Real dataset size: {len(real_dataset)}')
    print(f'Real dataset config: {real_data_config}')
    
    if len(real_dataset) == 0:
        print(f'❌ Real dataset is empty. Check if images exist in: {real_root}')
        if Path(real_root).exists():
            image_files = list(Path(real_root).glob('*.png')) + list(Path(real_root).glob('*.jpg'))
            print(f'Found {len(image_files)} image files in directory')
        else:
            print(f'❌ Directory does not exist: {real_root}')
    
    if len(fake_dataset) == 0:
        print('❌ No fake samples found!')
        return None
    if len(real_dataset) == 0:
        return None
    
    # Use batch size 100 like posterior_sample.py
    real_loader = DataLoader(real_dataset, batch_size=100, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)
    
    fid_score = calculate_fid(real_loader, fake_loader)
    
    # Save FID score
    with open(fid_dir / 'fid.txt', 'w') as f:
        f.write(f'FID Score: {fid_score.item():.4f}')
    
    print(f'FID Score: {fid_score.item():.4f}')
    return fid_score.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate FID for DAPS results')
    parser.add_argument('result_dir', help='Result directory path')
    parser.add_argument('--data', default=None, help='Dataset name (e.g., test-ffhq)')

    parser.add_argument('--data.start_id', dest='start_id', type=int, default=None, help='Dataset start ID')
    parser.add_argument('--data.end_id', dest='end_id', type=int, default=None, help='Dataset end ID')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for FID calculation')
    
    args = parser.parse_args()
    
    calculate_fid_for_result(
        args.result_dir,
        data=args.data,
        start_id=args.start_id,
        end_id=args.end_id,
        batch_size=args.batch_size
    )