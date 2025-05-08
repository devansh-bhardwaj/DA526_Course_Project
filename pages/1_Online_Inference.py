import torch 
import torch.nn as nn 
import torch.nn.functional as F
from models import PartialConv2d,PartialUNet,PatchDiscriminator,PDUNet,StyleGANModel,StyleGANDiscriminator,StyledGenerator,BinaryUNet
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import io

# Device and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def denorm(t):
    return (t + 1) / 2

# Load models
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

mask_predictor, styleCycleGAN, pconvunet, partial_unet = load_models()

# === Inference and separated outputs ===
def run_pipeline(masked_img_pil):
    masked_tensor = image_transform(masked_img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # Predict Binary Mask
        pred_mask = mask_predictor(masked_tensor)
        pred_mask_np = pred_mask[0, 0].cpu().numpy()
        pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
        _, binary_pred_mask = cv2.threshold(pred_mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_pred_mask_tensor = torch.tensor(binary_pred_mask).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        # Outputs from 3 models
        style_out = styleCycleGAN.G_masked_unmasked(masked_tensor)
        pconv_out = pconvunet(masked_tensor, binary_pred_mask_tensor)
        partial_out = partial_unet(masked_tensor, binary_pred_mask_tensor)

        # Reconstructed
        mask_3c = binary_pred_mask_tensor.expand_as(masked_tensor)
        recon_style = masked_tensor * (1 - mask_3c) + style_out * mask_3c
        recon_pconv = masked_tensor * (1 - mask_3c) + pconv_out * mask_3c
        recon_partial = masked_tensor * (1 - mask_3c) + partial_out * mask_3c

        # Denorm + convert to np arrays
        outputs = {
            "Masked Input": denorm(masked_tensor[0].cpu()).permute(1, 2, 0).numpy(),
            "Predicted Mask": binary_pred_mask_tensor[0].repeat(3, 1, 1).cpu().permute(1, 2, 0).numpy(),
            "StyleGAN Output": denorm(style_out[0].cpu()).permute(1, 2, 0).numpy(),
            "Reconstructed StyleGAN": denorm(recon_style[0].cpu()).permute(1, 2, 0).numpy(),
            "PConvGAN Output": denorm(pconv_out[0].cpu()).permute(1, 2, 0).numpy(),
            "Reconstructed PConvGAN": denorm(recon_pconv[0].cpu()).permute(1, 2, 0).numpy(),
            "Partial UNet Output": denorm(partial_out[0].cpu()).permute(1, 2, 0).numpy(),
            "Reconstructed PartialUNet": denorm(recon_partial[0].cpu()).permute(1, 2, 0).numpy(),
        }

        return outputs

# === Streamlit UI ===
# === Streamlit UI ===
st.title("Face Mask Inpainting Visualizer")

uploaded_file = st.file_uploader("Upload a face-masked image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Input Masked Image", width=400)

    if st.button("Run Inpainting"):
        with st.spinner("Processing..."):
            output_images = run_pipeline(input_image)

            # Arrange 2 images per row
            keys = list(output_images.keys())
            for i in range(0, len(keys), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(keys):
                        with cols[j]:
                            st.markdown(f"**{keys[i + j]}**")
                            st.image(output_images[keys[i + j]], width=300)
