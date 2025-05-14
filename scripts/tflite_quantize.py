import torch
import argparse
from torchvision import models
import ai_edge_torch
import torch.nn as nn
from ai_edge_torch.generative.quantize.quant_recipes import full_int8_dynamic_recipe

NUM_CLASSES = 6

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
model.load_state_dict(torch.load(path, weights_only=True))
sample_inputs = (torch.randn(1, 3, 128, 259),)

#TFLite Quantization: https://www.tensorflow.org/lite/performance/model_optimization
quant_config = full_int8_dynamic_recipe()

target = args.target_path
# Convert and serialize PyTorch model to a tflite flatbuffer.
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export(target)

ind = target.find('.tflite')
target_int8 = target[:ind] + '_int8.tflite'

edge_model_int8 = ai_edge_torch.convert(model.eval(), sample_inputs, quant_config)
edge_model_int8.export(target_int8)
