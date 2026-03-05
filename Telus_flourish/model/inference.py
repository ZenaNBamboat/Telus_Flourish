import torch
import numpy as np
import cv2
import os
import segmentation_models_pytorch as smp
import torch.serialization

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cpu"
IMG_SIZE = 256
MODEL_PATH = os.path.join("model", "best_model.pth")

# --------------------------------------------------
# PYTORCH 2.6+ SAFE GLOBALS (REQUIRED)
# --------------------------------------------------
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar
])

# --------------------------------------------------
# BUILD MODEL (MATCHES TRAINING)
# --------------------------------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.to(DEVICE)

# --------------------------------------------------
# LOAD CHECKPOINT (EXPLICIT OVERRIDES)
# --------------------------------------------------
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False   # 🔥 THIS IS CRITICAL
)

# Handle both formats safely
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("✅ ResNet-UNet loaded successfully")

# --------------------------------------------------
# INFERENCE FUNCTION
# --------------------------------------------------
def predict(image_rgb):
    img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    prob = output.squeeze().cpu().numpy()

    prob = cv2.resize(
        prob,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    confidence = float(np.max(prob))
    return prob, "Leaf Disease", confidence
