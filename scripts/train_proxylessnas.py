import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
from dataset import OwlSoundDataset
from tqdm import tqdm
import os

# Optional: Windows-only backend fix
torchaudio.set_audio_backend("soundfile")

# === Configuration ===
DATA_DIR = "../buowset"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
MODEL_PATH = "../models/proxylessnas_owl.pth"
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load metadata ===
metadata = pd.read_csv(META_FILE)

# Fixed folds: 0–2 for training, 3 for validation
train_df = metadata[metadata["fold"].isin([0, 1, 2])].reset_index(drop=True)
val_df = metadata[metadata["fold"] == 3].reset_index(drop=True)

# Datasets
train_dataset = OwlSoundDataset(train_df, AUDIO_DIR)
val_dataset = OwlSoundDataset(val_df, AUDIO_DIR)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load and configure proxyless_gpu ===
target_platform = "proxyless_mobile" # proxyless_gpu, proxyless_mobile, proxyless_mobile14 are also avaliable.
model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training loop ===
best_accuracy = 0.0
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
    for inputs, labels in train_loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    correct = 0
    total = 0

    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]")
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ New best model saved with accuracy: {best_accuracy:.4f}")

print("✅ Training complete.")
