# src/train.py

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


def compute_metrics(pred, target, threshold=0.5):
    pred   = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    pred_f   = pred.view(-1)
    target_f = target.view(-1)
    tp = (pred_f * target_f).sum()
    fp = (pred_f * (1 - target_f)).sum()
    fn = ((1 - pred_f) * target_f).sum()
    smooth = 1e-6
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    return {
        "dice":      dice.item(),
        "iou":       iou.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Eğitim", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Doğrulama", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            loss    = criterion(outputs, masks)
            total_loss += loss.item()
            batch_metrics = compute_metrics(outputs, masks)
            for k in total_metrics:
                total_metrics[k] += batch_metrics[k]
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


def train(model, train_loader, val_loader, config):
    device   = config["device"]
    epochs   = config["epochs"]
    save_dir = config["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    model     = model.to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(   # Adam → AdamW
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    writer    = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    best_dice = 0.0

    print(f"\n{'='*50}")
    print(f"Eğitim başlıyor — {epochs} epoch, cihaz: {device}")
    print(f"{'='*50}\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"IoU: {metrics['iou']:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"Metrics/{k}", v, epoch)

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"  ✓ Yeni en iyi model kaydedildi (Dice: {best_dice:.4f})")

    writer.close()
    print(f"\nEğitim tamamlandı! En iyi Dice: {best_dice:.4f}")