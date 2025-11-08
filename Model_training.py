# Model_training.py
"""
Train a small emotion classifier on FER (FER2013).
- By default uses torchvision.datasets.FER2013 (expects fer2013.csv in datasets/fer2013/)
- For a tiny local test you can use ImageFolder-style data in datasets/sample_faces/
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import FER2013
from tqdm import tqdm

# -------- CONFIG --------
DATA_ROOT = "datasets"
FER_ROOT = os.path.join(DATA_ROOT, "fer2013")  # if using FER2013 csv
SAMPLE_ROOT = os.path.join(DATA_ROOT, "samplefaces")  # optional tiny dataset (ImageFolder)
MODEL_OUT = "model.pth"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

# -------- MODEL --------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 24x24
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 12x12
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 6x6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------- TRANSFORMS & DATALOADERS --------
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def get_fer_loader(root, batch_size=BATCH_SIZE, val_split=0.1):
    # Use torchvision's FER2013 loader (requires the CSV to be present in root)
    dataset = FER2013(root=root, split='train', transform=transform, download=False)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def get_imagefolder_loader(root, batch_size=BATCH_SIZE, val_split=0.2):
    # For small quick tests: create subfolders per label inside sample_faces/
    dataset = datasets.ImageFolder(root=root, transform=transform)
    if len(dataset) < 2:
        raise RuntimeError("Not enough samples in sample_faces/ to form a dataset.")
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

# -------- TRAINING LOOP --------
def train(use_sample=False):
    model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if use_sample:
        print("Using ImageFolder samples from", SAMPLE_ROOT)
        train_loader, val_loader = get_imagefolder_loader(SAMPLE_ROOT)
    else:
        print("Using FER2013 dataset from", FER_ROOT)
        train_loader, val_loader = get_fer_loader(FER_ROOT)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for imgs, labels in loop:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=(running_loss / total), acc=(correct / total))

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0.0
        val_loss_avg = val_loss / val_total if val_total else 0.0
        print(f"Epoch {epoch}: val_loss={val_loss_avg:.4f}, val_acc={val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, MODEL_OUT)
            print(f"Saved best model to {MODEL_OUT}")

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    # For quick local testing with a few example images, run:
    #    python Model_training.py --sample
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use datasets/sample_faces (ImageFolder) instead of FER2013 CSV")
    args = parser.parse_args()
    train(use_sample=args.sample)
