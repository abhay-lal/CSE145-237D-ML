import torch
from torchvision import models
import os

FOLD = 4  # Use the best model fold
MODEL_PATH = f"../models/mobilenetv2_owl_fold{FOLD}.pth"
EXPORT_PATH = f"../exports/mobilenetv2.onnx"

NUM_CLASSES = 6
INPUT_SHAPE = (1, 3, 128, 128)  # adjust if your spectrogram is different

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Dummy input for shape
dummy_input = torch.randn(*INPUT_SHAPE)

# Export to ONNX
os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
torch.onnx.export(
    model, dummy_input, EXPORT_PATH,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=11
)

print(f"✅ Model exported to {EXPORT_PATH}")
