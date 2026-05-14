# 🔬 Derin Öğrenme Tabanlı Deri Lezyonu Bölütleme
**Attention U-Net (ResNet50 Encoder) | ISIC 2017+2018 | Dice: 0.813**

---

## 📌 Proje Hakkında

Bu proje, dermoskopik deri görüntülerinde lezyonları otomatik olarak tespit eden ve piksel bazında işaretleyen bir derin öğrenme sistemidir. Model bir deri fotoğrafına bakıp "lezyon tam olarak burada" diye işaretler.

**Ne yapar?**
- Deri görüntüsünü yükle → model lezyonu otomatik tespit eder
- Yeşil overlay ve kırmızı kontur ile lezyonu görselleştirir
- Lezyon alanı ve güven skoru hesaplar
- Ollama + Llama 3.1 ile Türkçe yapay zeka yorumu üretir

---

## 📁 Klasör Yapısı

```
skin_segmentation/
├── data/
│   ├── images/          ← dermoskopi görüntüleri (.jpg)
│   └── masks/           ← ikili maske görüntüleri (.png)
├── src/
│   ├── __init__.py
│   ├── dataset.py       ← veri yükleme ve augmentation
│   ├── model.py         ← Attention U-Net mimarisi
│   ├── train.py         ← eğitim döngüsü
│   └── visualize.py     ← tahmin görselleştirme
├── app/
│   └── app.py           ← Gradio kullanıcı arayüzü
├── checkpoints/
│   └── best_model.pth   ← eğitilmiş model ağırlıkları
├── results/             ← görselleştirme çıktıları
└── run.py               ← eğitimi başlatan ana dosya
```

---

## ⚙️ Kurulum

### 1. Gereksinimler

- Python 3.10+
- NVIDIA GPU (önerilen: 4GB+ VRAM)
- CUDA 12.x

### 2. Sanal ortam oluştur

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Kütüphaneleri kur

```bash
# PyTorch (CUDA 12.4 için)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Diğer kütüphaneler
pip install segmentation-models-pytorch albumentations opencv-python matplotlib scikit-learn tqdm tensorboard gradio ollama
```

### 4. Ollama kur (LLM yorumu için)

( https://ollama.com/download ) adresinden Ollama'yı indir ve kur, ardından:

```bash
ollama pull llama3.1:8b
```

---

## 📊 Veri Seti

ISIC 2018 ve ISIC 2017 veri setleri kullanılmıştır.

**İndirme adresi:** (https://challenge.isic-archive.com/data/#2018)

İndirilecek dosyalar (Task 1):
| Dosya | Boyut | Nereye |
|---|---|---|
| Training Input (2018) | ~10.4 GB | `data/images/` |
| Training Ground Truth (2018) | ~26 MB | `data/masks/` |
| Training Data (2017) | ~5 GB | `data/images/` |
| Training Ground Truth (2017) | ~86 MB | `data/masks/` |

Zip'leri açtıktan sonra klasör yapısı şöyle olmalı:
```
data/
├── images/   → ISIC_0024306.jpg, ISIC_0000000.jpg, ...
└── masks/    → ISIC_0024306_segmentation.png, ...
```

---

## 🚀 Kullanım

### Arayüzü Çalıştır (Hazır Model ile)

`checkpoints/best_model.pth` dosyası mevcutsa doğrudan arayüzü başlatabilirsiniz:

```bash
cd C:\skin_segmentation
python app/app.py
```

Tarayıcıda (http://127.0.0.1:7860) adresini aç.

**Kullanım:**
1. "Deri Görüntüsü Yükle" alanına bir dermoskopi fotoğrafı yükle
2. "Analiz Et" butonuna bas
3. Sonuçları gör: overlay, maske, güven skoru ve LLM yorumu

---

### Modeli Eğit (Opsiyonel)

Veri seti hazırsa eğitimi başlatmak için:

```bash
python run.py
```

Eğitim parametreleri `run.py` içinde ayarlanabilir:

```python
config = {
    "batch_size": 8,      # GPU belleğine göre azalt/artır
    "img_size":   256,    # görüntü boyutu
    "epochs":     50,     # epoch sayısı
    "lr":         1e-4,   # öğrenme hızı
}
```

> ⚠️ GPU'ya göre `batch_size` ayarla: 4GB VRAM için 4-8, 8GB+ için 16-32 önerilir.

Eğitim tamamlandığında en iyi model `checkpoints/best_model.pth` olarak kaydedilir.

---

### Tahminleri Görselleştir

```bash
python -m src.visualize
```

5 örnek görüntü için orijinal / gerçek maske / tahmin karşılaştırması `results/predictions.png` olarak kaydedilir.

---

## 📈 Model Performansı

| Metrik | Değer |
|---|---|
| **Dice** | **0.813** |
| **IoU** | **0.687** |
| Precision | 0.841 |
| Recall | 0.798 |

### Literatür Karşılaştırması

| Yöntem | Dice | Yıl |
|---|---|---|
| U-Net | 0.647 | 2015 |
| Attention U-Net | 0.748 | 2018 |
| TransUNet | 0.801 | 2021 |
| Swin-UNet | 0.812 | 2022 |
| **Önerilen Yöntem** | **0.813** | **2026** |

---

## 🏗️ Model Mimarisi

```
Girdi (384×384×3)
       ↓
   ENCODER (ResNet50 — ImageNet pretrained)
   Her seviyede boyut yarıya iner, kanal sayısı artar
       ↓
   BOTTLENECK
       ↓
   DECODER (4 seviye)
   Her seviyede:
     1. Transpoze konvolüsyon ile büyüt
     2. Attention Gate → skip connection'ı filtrele
     3. Filtreli bilgiyi birleştir
       ↓
   Çıktı Maskesi (384×384×1)
   Sigmoid → 0.5 eşiği → ikili maske
```

**Kayıp Fonksiyonu:** `L = 0.4 * BCE + 0.4 * Dice + 0.2 * Boundary`

**Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)

**LR Scheduler:** Cosine Annealing

---

## 🤖 LLM Entegrasyonu

Sistem Ollama + Llama 3.1:8b ile yerel yapay zeka yorumu üretir:
- İnternet bağlantısı gerekmez
- Türkçe genel değerlendirme
- Risk düzeyi: Düşük / Orta / Yüksek
- Klinik öneri

---
