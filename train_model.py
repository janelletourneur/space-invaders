import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset_fullframe"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODEL_PATH = PROJECT_DIR / "gesture_model.pth"

CLASSES = ["NONE", "LEFT", "RIGHT", "FIRE", "ENTER", "QUIT"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

class GestureCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def check_dataset():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable : {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable : {VAL_DIR}")

    for split_dir in [TRAIN_DIR, VAL_DIR]:
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Dossier de classe introuvable : {cls_dir}")


def count_images(split_dir: Path):
    counts = {}
    total = 0
    for cls in CLASSES:
        cls_dir = split_dir / cls
        n = len([p for p in cls_dir.iterdir() if p.is_file()])
        counts[cls] = n
        total += n
    return counts, total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    check_dataset()

    train_counts, train_total = count_images(TRAIN_DIR)
    val_counts, val_total = count_images(VAL_DIR)

    print("=== Vérification dataset ===")
    print(f"Train total: {train_total}")
    for cls in CLASSES:
        print(f"  {cls:<5} : {train_counts[cls]}")
    print(f"Val total:   {val_total}")
    for cls in CLASSES:
        print(f"  {cls:<5} : {val_counts[cls]}")
    print()

    if train_total == 0 or val_total == 0:
        raise RuntimeError("Le dataset est vide. Lance d'abord collect_dataset.py.")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    print("Classes détectées par ImageFolder:", train_dataset.classes)
    if train_dataset.classes != CLASSES:
        print("[ATTENTION] L'ordre des classes détecté n'est pas celui attendu.")
        print("Classes attendues:", CLASSES)
        print("On continue quand même, mais il faudra garder exactement le même ordre au moment de l'inférence.")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GestureCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = -1.0

    print("=== Début entraînement ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> meilleur modèle sauvegardé dans : {MODEL_PATH}")

    print()
    print("=== Fin entraînement ===")
    print(f"Meilleure val_acc : {best_val_acc:.4f}")
    print(f"Fichier modèle créé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
