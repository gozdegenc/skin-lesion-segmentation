# src/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(phase, img_size=256):
    if phase == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=False)


class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].unsqueeze(0)

        return image, mask


def get_dataloaders(data_dir, batch_size=8, img_size=256, seed=42):
    image_dir = os.path.abspath(os.path.join(data_dir, "images"))
    mask_dir  = os.path.abspath(os.path.join(data_dir, "masks"))

    mask_map = {}
    for f in os.listdir(mask_dir):
        if f.endswith("_segmentation.png"):
            img_id = f.replace("_segmentation.png", "")
            mask_map[img_id] = os.path.join(mask_dir, f)

    image_paths = []
    mask_paths  = []

    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith(".jpg"):
            continue
        img_id = img_file.replace(".jpg", "")
        if img_id in mask_map:
            image_paths.append(os.path.join(image_dir, img_file))
            mask_paths.append(mask_map[img_id])

   # Çift kontrolü — bozuk veya okunamayan dosyaları ele
    valid_image_paths = []
    valid_mask_paths  = []

    for ip, mp in zip(image_paths, mask_paths):
        if not os.path.exists(mp):
            continue
        # Gerçekten okunabilir mi kontrol et
        img  = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"UYARI: Bozuk dosya atlandı → {os.path.basename(ip)}")
            continue
        valid_image_paths.append(ip)
        valid_mask_paths.append(mp)

    image_paths = valid_image_paths
    mask_paths  = valid_mask_paths

    print(f"Geçerli çift: {len(image_paths)}")

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=seed
    )

    train_ds = SkinLesionDataset(train_imgs, train_masks, get_transforms("train", img_size))
    val_ds   = SkinLesionDataset(val_imgs,   val_masks,   get_transforms("val",   img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Eğitim: {len(train_ds)} | Doğrulama: {len(val_ds)}")
    return train_loader, val_loader