# Required libraries:
# pip install torch torchvision tensorflow==2.13.0 onnx==1.14.1 onnx-tf==1.10.0
# pip install tensorflow_probability==0.20.0

import os
import sys
import torch
import torch.nn as nn
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

if not os.path.exists("proxylessnas"):
    os.system("git clone https://github.com/mit-han-lab/proxylessnas.git")

sys.path.append("proxylessnas")

from proxyless_nas.model_zoo import proxyless_mobile

model = proxyless_mobile()
model.classifier.linear = nn.Linear(model.classifier.linear.in_features, 6)
model.first_conv.conv = nn.Conv2d(
    in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False
)

state_dict = torch.load("proxylessnas_owl.pth", map_location='cpu')
if "classifier.weight" in state_dict and "classifier.bias" in state_dict:
    state_dict["classifier.linear.weight"] = state_dict.pop("classifier.weight")
    state_dict["classifier.linear.bias"] = state_dict.pop("classifier.bias")

model.load_state_dict(state_dict)
model.eval()
print("Loaded proxylessnas_owl.pth with 1-channel input and 6-class output.")

dummy_input = torch.randn(1, 1, 64, 258)
torch.onnx.export(
    model, dummy_input, "proxyless_buow.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11
)
print("Exported to ONNX: proxyless_buow.onnx")

onnx_model = onnx.load("proxyless_buow.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("proxyless_buow_tf")
print("Converted to TensorFlow SavedModel at ./proxyless_buow_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("proxyless_buow_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("proxyless_buow.tflite", "wb") as f:
    f.write(tflite_model)

print("Converted to TFLite: proxyless_buow.tflite")

os.system("xxd -i proxyless_buow.tflite > buow_model_data.h")
print("Created buow_model_data.h")
