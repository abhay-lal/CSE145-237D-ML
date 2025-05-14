import tensorflow as tf
from onnx_tf.backend import prepare
import onnx
import os

ONNX_PATH = "../exports/mobilenetv2.onnx"
TF_DIR = "../exports/mobilenetv2_tf"
TFLITE_PATH = "../exports/mobilenetv2.tflite"

# Load ONNX model and convert to TensorFlow SavedModel
onnx_model = onnx.load(ONNX_PATH)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(TF_DIR)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(TF_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables quantization (optional)

tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved to {TFLITE_PATH}")
