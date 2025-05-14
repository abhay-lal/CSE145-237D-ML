import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from dataset import OwlSoundDataset
import os

# --- Config ---
DATA_DIR = "../buowset"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
BATCH_SIZE = 128
NUM_CLASSES = 6
FOLD = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
metadata = pd.read_csv(META_FILE)
test_df = metadata[metadata["fold"] == FOLD].reset_index(drop=True)
test_dataset = OwlSoundDataset(test_df, AUDIO_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(model, name):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = sum([p == l for p, l in zip(preds, labels)]) / len(labels)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(labels, preds))

    # Plot confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# --- MobileNetV2 ---
mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, NUM_CLASSES)
mobilenet.load_state_dict(torch.load("../models/mobilenetv2_owl.pth", map_location=DEVICE))
mobilenet.to(DEVICE)
evaluate(mobilenet, "MobileNetV2")

# --- ProxylessNAS ---
proxyless = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=True)
proxyless.classifier = nn.Linear(proxyless.classifier.in_features, NUM_CLASSES)
proxyless.load_state_dict(torch.load("../models/proxylessnas_owl.pth", map_location=DEVICE))
proxyless.to(DEVICE)
evaluate(proxyless, "ProxylessNAS")
