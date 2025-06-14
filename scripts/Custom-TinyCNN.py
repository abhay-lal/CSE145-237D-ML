# -*- coding: utf-8 -*-
"""Custom-TinyCNN.ipynb

Automatically generated by Colab - Abhay Lal.

Original file is located at
    https://colab.research.google.com/drive/1UBXjuzb1xXvMCT4kOXPcCh9k3wFhK6um
"""

import requests

# This is the direct download link constructed from your SharePoint link
url = 'https://ucsdcloud-my.sharepoint.com/personal/ablal_ucsd_edu/_layouts/15/download.aspx?share=EavH0dkbioFKj6EWSgzL61IBYTYRdDFFowXfcQWsF-wJdQ'

output_file = 'data.zip'

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded successfully to '{output_file}'")
else:
    print(f"Failed to download. Status code: {response.status_code}")

!unzip data.zip

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Config
ROOT_DIR = "precomputed_mels"
VAL_FOLD = 0
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 6
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class BUOWMelDataset(Dataset):
    def __init__(self, root_dir, val_fold=0, train=True):
        self.samples = []
        fold_prefix = f"fold{val_fold}"
        for fold_name in os.listdir(root_dir):
            if (train and fold_name != fold_prefix) or (not train and fold_name == fold_prefix):
                fold_path = os.path.join(root_dir, fold_name)
                for fname in os.listdir(fold_path):
                    if fname.endswith(".pt"):
                        self.samples.append(os.path.join(fold_path, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx])
        mel = sample["mel"]          # shape: (1, 64, 258)
        label = sample["label"]
        return mel, label

# Lightweight CNN
class TinyAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,8,32,129)

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,16,64)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Training pipeline
def train_model():
    train_dataset = BUOWMelDataset(ROOT_DIR, val_fold=VAL_FOLD, train=True)
    val_dataset = BUOWMelDataset(ROOT_DIR, val_fold=VAL_FOLD, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TinyAudioCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_acc_list, val_acc_list, train_loss_list = [], [], []
    precision_list, recall_list, f1_list = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_acc_list.append(val_acc)

        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        print(f"Val Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save model
    torch.save(model.state_dict(), "buow_tinycnn.pth")
    print("Saved model to buow_tinycnn.pth")

    # Save metrics
    metrics_df = pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "train_loss": train_loss_list,
        "train_accuracy": train_acc_list,
        "val_accuracy": val_acc_list,
        "precision": precision_list,
        "recall": recall_list,
        "f1_score": f1_list
    })
    metrics_df.to_csv("training_metrics.csv", index=False)
    print("Saved training metrics to training_metrics.csv")

    # Plots
    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_accuracy"], label="Train Accuracy")
    plt.plot(metrics_df["epoch"], metrics_df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["precision"], label="Precision")
    plt.plot(metrics_df["epoch"], metrics_df["recall"], label="Recall")
    plt.plot(metrics_df["epoch"], metrics_df["f1_score"], label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 over Epochs")
    plt.legend()
    plt.savefig("precision_recall_f1.png")
    plt.close()

    print("Saved plots: train_loss.png, accuracy_curve.png, precision_recall_f1.png")

if __name__ == "__main__":
    train_model()

