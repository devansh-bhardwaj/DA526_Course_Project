import streamlit as st
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import cv2
from glob import glob
from models import *
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("Model Comparison on Dataset Samples")

# Device and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
mask_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
def denorm(t): return (t + 1) / 2

@st.cache_resource
def load_models():
    mask_predictor = BinaryUNet(in_channels=3, out_channels=1, base_channels=64).to(device)
    mask_predictor.load_state_dict(torch.load('binaryMask_best_model.pth', map_location=device)['unet_state_dict'])
    mask_predictor.eval()
    
    styleCycleGAN = StyleGANModel().to(device)
    styleCycleGAN.load_state_dict(torch.load('styleCycleGAN_best_model.pth', map_location=device))
    styleCycleGAN.eval()
    
    pconvunet = PDUNet().to(device)
    pconvunet.load_state_dict(torch.load('PartialUNet_best_model.pth', map_location=device)['unet_state_dict'])
    pconvunet.eval()
    
    partial_unet = PartialUNet(in_channels=3, out_channels=3).to(device)
    partial_unet.load_state_dict(torch.load('PConvGAN_best_model.pth', map_location=device)['generator_state_dict'])
    partial_unet.eval()

    return mask_predictor, styleCycleGAN, pconvunet, partial_unet

# Load all models
mask_predictor, styleCycleGAN, pconvunet, partial_unet = load_models()

# Image folders
masked_dir = "dataset/masked_images"
original_dir = "dataset/original_images"
binary_mask_dir = "dataset/binary_masks"

@st.cache_data
def get_comparison_grid(num_samples=10):
    from tqdm import tqdm

    masked_images = glob(os.path.join(masked_dir, "*.jpg"))[:num_samples]
    pairs = []
    for masked_path in masked_images:
        fname = os.path.basename(masked_path).split("_")[0] + ".jpg"
        original_path = os.path.join(original_dir, fname)
        mask_path = os.path.join(binary_mask_dir, os.path.basename(masked_path).replace(".jpg", "_binary.jpg"))
        if os.path.exists(original_path) and os.path.exists(mask_path):
            pairs.append((masked_path, original_path, mask_path))

    vis_rows = []
    for masked_path, original_path, gt_mask_path in tqdm(pairs, desc="Processing Samples"):
        # Load masked image
        masked_tensor = image_transform(Image.open(masked_path).convert("RGB")).unsqueeze(0).to(device)

        # Ground Truth Mask
        gt_mask_tensor = mask_transform(Image.open(gt_mask_path).convert("L")).unsqueeze(0).to(device)

        # Predict Binary Mask
        with torch.no_grad():
            pred_mask = mask_predictor(masked_tensor)
            pred_mask_np = pred_mask[0, 0].cpu().numpy()
            pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
            _, binary_pred_mask = cv2.threshold(pred_mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_pred_mask_tensor = torch.tensor(binary_pred_mask).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        # Model Outputs
        with torch.no_grad():
            style_out = styleCycleGAN.G_masked_unmasked(masked_tensor)
            pconv_out = pconvunet(masked_tensor, binary_pred_mask_tensor)
            partial_out = partial_unet(masked_tensor, binary_pred_mask_tensor)

        # Reconstructions
        mask_3c = binary_pred_mask_tensor.expand_as(masked_tensor)
        recon_style = masked_tensor * (1 - mask_3c) + style_out * mask_3c
        recon_pconv = masked_tensor * (1 - mask_3c) + pconv_out * mask_3c
        recon_partial = masked_tensor * (1 - mask_3c) + partial_out * mask_3c

        # Original
        original_tensor = transforms.ToTensor()(Image.open(original_path).convert("RGB").resize((64, 64))).unsqueeze(0).to(device)
        original_tensor = (original_tensor - 0.5) / 0.5

        # Row of 10 images
        vis_row = torch.stack([
            denorm(masked_tensor[0].cpu()),
            gt_mask_tensor[0].repeat(3, 1, 1).cpu(),
            binary_pred_mask_tensor[0].repeat(3, 1, 1).cpu(),
            denorm(style_out[0].cpu()),
            denorm(pconv_out[0].cpu()),
            denorm(partial_out[0].cpu()),
            denorm(recon_style[0].cpu()),
            denorm(recon_pconv[0].cpu()),
            denorm(recon_partial[0].cpu()),
            denorm(original_tensor[0].cpu())
        ], dim=0)

        vis_rows.append(vis_row)

    vis_tensor = torch.cat(vis_rows, dim=0)
    final_grid = make_grid(vis_tensor, nrow=10, padding=2)
    return final_grid.permute(1, 2, 0).numpy()

# Show image grid
grid_np = get_comparison_grid(num_samples=10)
st.image(grid_np, caption="Masked | GT Mask | Pred Mask | StyleGAN | PConvGAN | PartialUNet | Recon Style | Recon PConv | Recon Partial | Original", use_container_width=True)
