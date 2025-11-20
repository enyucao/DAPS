# DAPS Download Script for Windows PowerShell

# Create directories
New-Item -ItemType Directory -Force -Path "checkpoints"
New-Item -ItemType Directory -Force -Path "dataset"
New-Item -ItemType Directory -Force -Path "forward_operator\bkse\experiments\pretrained"

# FFHQ 256 DDPM
Write-Host "Downloading FFHQ 256 DDPM model..."
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O checkpoints/ffhq256.pt

# ImageNet 256 DDPM
Write-Host "Downloading ImageNet 256 DDPM model..."
gdown https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O checkpoints/imagenet256.pt

# FFHQ 256 LDM
Write-Host "Downloading FFHQ 256 LDM model..."
Invoke-WebRequest -Uri "https://ommer-lab.com/files/latent-diffusion/ffhq.zip" -OutFile "checkpoints/ffhq.zip"
Expand-Archive -Path "checkpoints/ffhq.zip" -DestinationPath "checkpoints" -Force
Move-Item -Path "checkpoints/model.ckpt" -Destination "checkpoints/ldm_ffhq256.pt" -Force
Remove-Item -Path "checkpoints/ffhq.zip" -Force

# ImageNet 256 LDM
Write-Host "Downloading ImageNet 256 LDM model..."
Invoke-WebRequest -Uri "https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt" -OutFile "checkpoints/ldm_imagenet256.pt"

# Nonlinear blur model
Write-Host "Downloading nonlinear blur model..."
gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O forward_operator/bkse/experiments/pretrained/GOPRO_wVAE.pth

# FFHQ test dataset
Write-Host "Downloading FFHQ test dataset..."
gdown https://drive.google.com/uc?id=1IzbnLWPpuIw6Z2E4IKrRByh6ihDE5QLO -O dataset/test-ffhq.zip
Expand-Archive -Path "dataset/test-ffhq.zip" -DestinationPath "dataset" -Force
Remove-Item -Path "dataset/test-ffhq.zip" -Force

# ImageNet test dataset
Write-Host "Downloading ImageNet test dataset..."
gdown https://drive.google.com/uc?id=1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs -O dataset/test-imagenet.zip
Expand-Archive -Path "dataset/test-imagenet.zip" -DestinationPath "dataset" -Force
Remove-Item -Path "dataset/test-imagenet.zip" -Force

Write-Host "Download completed!"