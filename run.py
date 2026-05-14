# run.py

import torch
import segmentation_models_pytorch as smp
from src.dataset import get_dataloaders
from src.train   import train

config = {
    "data_dir":     "data",
    "batch_size":   8,       # 4'ten 8'e çıkardık
    "img_size":     256,     # 224'ten 256'ya çıkardık
    "epochs":       30,      # 50'den 30'a düşürdük (overfitting önlemek için)
    "lr":           3e-4,    # öğrenme hızını artırdık
    "weight_decay": 1e-4,    # regularization artırdık
    "save_dir":     "checkpoints",
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(
        data_dir   = config["data_dir"],
        batch_size = config["batch_size"],
        img_size   = config["img_size"],
    )

    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
        decoder_use_norm = True,
    )

    train(model, train_loader, val_loader, config)