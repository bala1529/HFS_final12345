from pathlib import Path

import torch
import timm

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Xception backbone (ImageNet-pretrained)
print("Downloading Xception (timm/xception41.tf_in1k)...")
xception = timm.create_model("xception41.tf_in1k", pretrained=True, num_classes=2)
x_path = MODELS_DIR / "xception_model.pth"
torch.save(xception.state_dict(), x_path)
print(f"Saved Xception weights to {x_path}")

# 2) ConvNeXt backbone (ImageNet-pretrained)
print("Downloading ConvNeXt (convnext_tiny.fb_in1k)...")
convnext = timm.create_model("convnext_tiny.fb_in1k", pretrained=True, num_classes=2)
c_path = MODELS_DIR / "convnext_model.pth"
torch.save(convnext.state_dict(), c_path)
print(f"Saved ConvNeXt weights to {c_path}")

print("Done.")