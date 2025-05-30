import os
import sys
import torch
import torch.nn as nn
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare  

class TinyAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

NUM_CLASSES = 6
model = TinyAudioCNN(num_classes=NUM_CLASSES)
state_dict = torch.load("buow_tinycnn.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
print("Loaded TinyAudioCNN model with 1-channel input and 6-class output.")

dummy_input = torch.randn(1, 1, 64, 258)
torch.onnx.export(
    model, dummy_input, "buow_tinycnn.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11
)
print("Exported to ONNX: buow_tinycnn.onnx")

onnx_model = onnx.load("buow_tinycnn.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("buow_tinycnn")
print("Converted to TensorFlow SavedModel at ./buow_tinycnn")

converter = tf.lite.TFLiteConverter.from_saved_model("buow_tinycnn")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
tflite_model = converter.convert()

with open("buow_tinycnn.tflite", "wb") as f:
    f.write(tflite_model)

print("Converted to TFLite: buow_tinycnn.tflite")

os.system("xxd -i buow_tinycnn.tflite > buow_tinycnn.h")
print("Created buow_tinycnn.h")