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

# Optional: Windows-only backend config
torchaudio.set_audio_backend("soundfile")

# Setup
DATA_DIR = "../buowset"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
NUM_CLASSES = 6
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda")
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load metadata.csv file
metadata = pd.read_csv(META_FILE)

# K-Fold Training
for fold in range(5):
    print(f"\n=== Fold {fold} ===")

    # Split dataset
    train_df = metadata[metadata["fold"] != fold].reset_index(drop=True)
    val_df = metadata[metadata["fold"] == fold].reset_index(drop=True)

    train_dataset = OwlSoundDataset(train_df, AUDIO_DIR)
    val_dataset = OwlSoundDataset(val_df, AUDIO_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Define model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Fold {fold}, Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        val_loop = tqdm(val_loader, desc=f"Fold {fold} | Epoch {epoch+1} [Validation]")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Fold {fold}, Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}")

        # Save best model per fold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = os.path.join(MODEL_DIR, f"mobilenetv2_owl_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved for fold {fold} with accuracy: {best_accuracy:.4f}")

print("\n✅ 5-Fold Training Complete.")
