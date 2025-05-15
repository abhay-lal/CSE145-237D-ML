import torch
import argparse
from torchvision import models
import ai_edge_torch
import torch.nn as nn
from ai_edge_torch.generative.quantize.quant_recipes import full_int8_dynamic_recipe
import os

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.quantize.quant_config import QuantConfig

from torch.quantization import get_default_qconfig, prepare, convert

NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('model_type', help='ProxylessNAS or MobileNetV2')
parser.add_argument('model_path', help='Path to the model to convert to TFLite')
parser.add_argument('target_path', help='Path to store .tflite file')

args = parser.parse_args()

if args.model_type == 'ProxylessNAS':
    model = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
elif args.model_type == 'MobileNetV2':
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
else:
    print('Please specify -model_type as \'ProxylessNAS\' or \'MobileNetV2\'')
    print('Defaulting to ProxylessNAS')
    model = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

path = args.model_path

print("Loading PyTorch model to convert")
model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))

print("Finished loading PyTorch model")

sample_inputs = (torch.randn(1, 3, 128, 241),)

target = args.target_path
# Convert and serialize PyTorch model to a tflite flatbuffer.

print("Converting PyTorch Model to TFLite without Quantization (float32)")
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export(target)
print(f"Finished converting, exported to {target}")

print("Converting PyTorch Model to TFLite with Quantization (int8)")
ind = target.find('.tflite')
target_int8 = target[:ind] + '_int8.tflite'

quant_config = full_int8_dynamic_recipe()

pt2e_drq_model = ai_edge_torch.convert(model, sample_inputs, quant_config=quant_config)
pt2e_drq_model.export(target_int8)

print(f"Finished converting, exported to {target_int8}")

size_mb = os.path.getsize(target) / (2 ** 20) #bytes to Mb
size_mb_int8 = os.path.getsize(target_int8) / (2 ** 20)
print(f"Model size without quantization (float32): {size_mb}")
print(f"Model size with quantization (int8): {size_mb_int8}")



# ====== PT2E METHOD =======
# pt2e_quantizer = PT2EQuantizer().set_global(
#     get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
# )

# pt2e_torch_model = capture_pre_autograd_graph(model.eval(), sample_inputs)
# pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)

# # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
# pt2e_torch_model(*sample_inputs)

# # Convert the prepared model to a quantized model
# pt2e_torch_model = convert_pt2e(pt2e_torch_model.eval(), fold_quantize=False)

# # Convert to an ai_edge_torch model