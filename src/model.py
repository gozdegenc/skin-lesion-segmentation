# src/model.py

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# YARDIMCI BLOKLAR
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    İki kez: Konvolüsyon → BatchNorm → ReLU
    Bu blok encoder ve decoder'da tekrar tekrar kullanılır.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate: Skip connection'daki hangi bölgelerin önemli
    olduğunu öğrenir ve önemsiz bölgeleri bastırır.

    g  = decoder'dan gelen sinyal (ne aradığımızı bilir)
    x  = encoder'dan gelen skip connection (detay bilgisi taşır)
    """
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()

        # Decoder sinyalini dönüştür
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Encoder sinyalini dönüştür
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # İkisini birleştirip dikkat haritası üret (0-1 arası)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),   # her piksele 0-1 arası önem skoru ver
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)   # ikisini topla
        psi = self.psi(psi)         # dikkat haritası (0=önemsiz, 1=önemli)
        return x * psi              # encoder çıktısını filtrele


# ─────────────────────────────────────────────
# ANA MİMARİ: ATTENTION U-NET
# ─────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net — 4 seviyeli encoder/decoder + attention gate'ler.

    Encoder: görüntüyü küçülterek soyut özellikler çıkarır
    Bottleneck: en derin, en soyut temsil
    Decoder: küçük özellik haritasını tekrar büyütür
    Skip connections: encoder detaylarını decoder'a taşır
    Attention gates: bu detayları akıllıca filtreler
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # ── ENCODER (aşağı iniş) ──────────────────
        self.enc1 = DoubleConv(in_channels, features[0])   # 3   → 64
        self.enc2 = DoubleConv(features[0], features[1])   # 64  → 128
        self.enc3 = DoubleConv(features[1], features[2])   # 128 → 256
        self.enc4 = DoubleConv(features[2], features[3])   # 256 → 512

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # boyutu yarıya indir

        # ── BOTTLENECK (en dip) ───────────────────
        self.bottleneck = DoubleConv(features[3], features[3] * 2)  # 512 → 1024

        # ── DECODER (yukarı çıkış) ────────────────
        # Her seviyede: büyüt → attention → birleştir → çift konv

        self.up4    = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.att4   = AttentionGate(features[3], features[3], features[3]//2)
        self.dec4   = DoubleConv(features[3]*2, features[3])   # 1024 → 512

        self.up3    = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3   = AttentionGate(features[2], features[2], features[2]//2)
        self.dec3   = DoubleConv(features[2]*2, features[2])   # 512 → 256

        self.up2    = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2   = AttentionGate(features[1], features[1], features[1]//2)
        self.dec2   = DoubleConv(features[1]*2, features[1])   # 256 → 128

        self.up1    = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.att1   = AttentionGate(features[0], features[0], features[0]//2)
        self.dec1   = DoubleConv(features[0]*2, features[0])   # 128 → 64

        # ── ÇIKIŞ ────────────────────────────────
        # 64 kanalı → 1 kanala indir (lezyon var mı yok mu?)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encoder ──
        e1 = self.enc1(x)           # 256×256, 64 kanal
        e2 = self.enc2(self.pool(e1))  # 128×128, 128 kanal
        e3 = self.enc3(self.pool(e2))  # 64×64,  256 kanal
        e4 = self.enc4(self.pool(e3))  # 32×32,  512 kanal

        # ── Bottleneck ──
        b  = self.bottleneck(self.pool(e4))  # 16×16, 1024 kanal

        # ── Decoder (her seferinde: büyüt → attention → concat → conv) ──
        d4 = self.up4(b)                        # 32×32,  512
        e4 = self.att4(g=d4, x=e4)             # attention filtrele
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # birleştir + conv

        d3 = self.up3(d4)                       # 64×64,  256
        e3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)                       # 128×128, 128
        e2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                       # 256×256, 64
        e1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # ── Çıktı ──
        return self.final_conv(d1)  # 256×256, 1 kanal (ham skor)