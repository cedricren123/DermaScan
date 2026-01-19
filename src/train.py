import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataset import Ham10000Binary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_loaders(csv_path: str, images_dir: str, batch_size: int = 16):
    # training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(260, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # val size must match model input size for EfficientNet-B2
    val_tf = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Building dataset just for labels (stratified split)
    base = Ham10000Binary(csv_path=csv_path, images_dir=images_dir, transform=None)
    y = base.df["label"].values
    idx = np.arange(len(base))

    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )

    # Real datasets with transforms
    train_ds = Ham10000Binary(csv_path=csv_path, images_dir=images_dir, transform=train_tf)
    val_ds   = Ham10000Binary(csv_path=csv_path, images_dir=images_dir, transform=val_tf)

    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def build_model():
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, preds, prob1 = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        ys.extend(y.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
        prob1.extend(probs[:, 1].cpu().numpy().tolist())

    acc = accuracy_score(ys, preds)
    try:
        auc = roc_auc_score(ys, prob1)
    except Exception:
        auc = float("nan")
    return acc, auc

def train(csv_path: str, images_dir: str, epochs: int = 15, lr: float = 1e-4, batch_size: int = 16):
    # Loaders
    train_loader, val_loader = make_loaders(csv_path, images_dir, batch_size=batch_size)

    # Model
    model = build_model().to(DEVICE)

    # Weighted loss to address class imbalance
    base = Ham10000Binary(csv_path=csv_path, images_dir=images_dir, transform=None)
    counts = base.df["label"].value_counts().to_dict()
    w0 = 1.0 / counts[0]
    w1 = 1.0 / counts[1]
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    best_auc = -1.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_auc = evaluate(model, val_loader)
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={total_loss/len(train_loader):.4f} | "
            f"val_acc={val_acc:.3f} | val_auc={val_auc:.3f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "models/best.pt")
            print("  ✅ saved models/best.pt")

if __name__ == "__main__":
    CSV_PATH = "data/metadata.csv"
    IMAGES_DIR = "data/images"

    train(CSV_PATH, IMAGES_DIR, epochs=15, lr=1e-4, batch_size=16)
