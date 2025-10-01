import torch
import numpy as np
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms
import os

# ---- Step 1: Load 2D MRI slice ----
nii_path = "mri_slice.nii"   # replace with your file
save_path = "sr_mri.png"

# Load the 2D slice from NIfTI
slice2d = nib.load(nii_path).get_fdata().astype(np.float32)

# Normalize to [0,1]
slice2d = (slice2d - slice2d.min()) / (slice2d.max() - slice2d.min() + 1e-8)

# Convert grayscale -> 3-channel RGB
slice_rgb = np.stack([slice2d]*3, axis=-1)

# Convert to PIL Image
img = Image.fromarray((slice_rgb*255).astype(np.uint8))

# ---- Step 2: Preprocess ----
transform = transforms.Compose([
    transforms.ToTensor()
])
lr_tensor = transform(img).unsqueeze(0)  # shape: [1,3,H,W]

# ---- Step 3: Import ESRGAN model ----
from esrgan_pytorch.models import RRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = RRDBNet(
    in_channels=3, 
    out_channels=3, 
    channels=64, 
    growth_channels=32, 
    num_blocks=23, 
    scale_factor=4
).to(device)

# ---- Step 4: Load pretrained weights ----
# Download from https://github.com/Lornatang/ESRGAN-PyTorch/releases
model_path = "RRDB_ESRGAN_x4.pth.tar"  # put in same folder

checkpoint = torch.load(model_path, map_location=device)
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# ---- Step 5: Run inference ----
with torch.no_grad():
    sr_tensor = model(lr_tensor.to(device))

# ---- Step 6: Postprocess & Save ----
sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0,1)
sr_img = transforms.ToPILImage()(sr_tensor)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
sr_img.save(save_path)
print(f"Super-resolved MRI saved at {save_path}")
