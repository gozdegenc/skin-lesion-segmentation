# src/visualize.py

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from src.dataset import get_transforms

def visualize_predictions(model_path, data_dir, num_samples=5, img_size=384):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modeli yükle
    model = smp.Unet(
        encoder_name         = "resnet50",
        encoder_weights      = None,
        in_channels          = 3,
        classes              = 1,
        activation           = None,
        decoder_use_batchnorm= True,  # type: ignore[arg-type]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model yüklendi!")

    # Birkaç görüntü seç
    image_dir = os.path.join(data_dir, "images")
    mask_dir  = os.path.join(data_dir, "masks")
    transform = get_transforms("val", img_size)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    fig.suptitle("Orijinal | Gerçek Maske | Tahmin", fontsize=14, fontweight="bold")

    for i, img_file in enumerate(image_files):
        img_id    = img_file.replace(".jpg", "")
        mask_file = img_id + "_segmentation.png"

        img_path  = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            continue

        # Görüntüyü oku
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_gt   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gt is None:
            continue

        # Modele sok
        aug     = transform(image=image_rgb, mask=(mask_gt > 127).astype(np.float32))
        inp     = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

        pred_binary = (pred > 0.5).astype(np.uint8) * 255

        # Görselleştir
        axes[i, 0].imshow(cv2.resize(image_rgb, (img_size, img_size)))
        axes[i, 0].set_title("Orijinal Görüntü")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.resize(mask_gt, (img_size, img_size)), cmap="gray")
        axes[i, 1].set_title("Gerçek Maske")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_binary, cmap="gray")
        axes[i, 2].set_title(f"Tahmin")
        axes[i, 2].axis("off")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Görsel kaydedildi: {save_path}")


if __name__ == "__main__":
    visualize_predictions(
        model_path  = "checkpoints/best_model.pth",
        data_dir    = "data",
        num_samples = 5,
        img_size    = 256,
    )