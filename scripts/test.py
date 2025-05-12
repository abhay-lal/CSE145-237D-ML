import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from dataset import OwlSoundDataset
import os

# Testing model FOLD on the test set from FOLD since FOLD is trained on every other folds but the original FOLD

# Setup
FOLD = 1  # Test set fold
DATA_DIR = "../buowset"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
MODEL_PATH = f"../models/mobilenetv2_owl_fold{FOLD}.pth"
BATCH_SIZE = 128
NUM_CLASSES = 6
DEVICE = torch.device("cuda")

# Load metadata.csv and prepare the test set
metadata = pd.read_csv(META_FILE)
test_df = metadata[metadata["fold"] == FOLD].reset_index(drop=True)
test_dataset = OwlSoundDataset(test_df, AUDIO_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Evalutation
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy on Fold {FOLD}: {accuracy:.4f}")
