# app/app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import cv2
import numpy as np
import torch
import gradio as gr
from gradio import themes
import segmentation_models_pytorch as smp
from src.dataset import get_transforms
import ollama

# ── Modeli yükle ───────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name         = "resnet50",
    encoder_weights      = None,
    in_channels          = 3,
    classes              = 1,
    activation           = None,
    decoder_use_norm     = True,
).to(device)

model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth"),
    map_location=device
))
model.eval()
print(f"Model yüklendi! Cihaz: {device}")

transform = get_transforms("val", img_size=384)


# ── Tek görüntü tahmini ────────────────────────────
def predict_single(image_rgb):
    aug = transform(image=image_rgb, mask=np.zeros(image_rgb.shape[:2], dtype=np.float32))
    inp = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_binary = (prob > 0.5).astype(np.uint8)
    return prob, pred_binary


# ── Sonuçları görselleştir ─────────────────────────
def make_visuals(image_rgb, prob, pred_binary, img_size=384):
    image_resized = cv2.resize(image_rgb, (img_size, img_size))

    # Yeşil overlay
    overlay = image_resized.copy()
    overlay[pred_binary == 1] = (
        overlay[pred_binary == 1] * 0.4 + np.array([0, 220, 0]) * 0.6
    ).astype(np.uint8)

    # Lezyon sınırını kırmızı çiz
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    # Maske (siyah-beyaz)
    mask_rgb = cv2.cvtColor((pred_binary * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Güven skoru — lezyon bölgesindeki ortalama olasılık
    if pred_binary.sum() > 0:
        guven = float(prob[pred_binary == 1].mean()) * 100
    else:
        guven = 0.0

    return overlay, mask_rgb, guven


# ── Tek görüntü analizi ────────────────────────────
def analyze_single(image):
    if image is None:
        return None, None, "", 0

    image_rgb = image.astype(np.uint8)
    prob, pred_binary = predict_single(image_rgb)
    overlay, mask_rgb, guven = make_visuals(image_rgb, prob, pred_binary)

    lezyon_oran = (pred_binary.sum() / pred_binary.size) * 100
    tespit = "Lezyon tespit edildi ✅" if lezyon_oran > 1 else "Lezyon tespit edilemedi ❌"

    # Temel bilgiler
    temel_bilgi = (
        f"📊 Analiz Sonuçları\n\n"
        f"• {tespit}\n"
        f"• Lezyon alanı: %{lezyon_oran:.1f}\n"
        f"• Güven skoru: %{guven:.1f}\n\n"
    )

    # LLM yorumu
    try:
        prompt = f"""Sen deneyimli bir dermatoloji asistanısın ve yalnızca Türkçe konuşuyorsun. 
Hiçbir şekilde İngilizce, Çince veya başka bir dil kullanma.

Bir derin öğrenme modeli dermoskopik deri görüntüsünü analiz etti:
- Lezyon tespit edildi mi: {'Evet' if lezyon_oran > 1 else 'Hayır'}
- Lezyonun görüntü içindeki alanı: yüzde {lezyon_oran:.1f}
- Modelin güven skoru: yüzde {guven:.1f}

Lütfen şu başlıklar altında Türkçe yorum yap:

1. Genel Değerlendirme: Tespit edilen lezyonu kısaca değerlendir.
2. Risk Düzeyi: Lezyon alanı ve güven skoruna göre düşük/orta/yüksek risk belirt.
3. Öneri: Hastaya ne yapması gerektiğini söyle.

Son cümle mutlaka: "Kesin tanı için bir dermatologa başvurmanız önerilir." olsun.
Sadece Türkçe kullan."""

        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}]
        )
        llm_yorum = response["message"]["content"]
    except Exception as e:
        llm_yorum = "LLM yorumu alınamadı."

    bilgi = temel_bilgi + "🤖 Yapay Zeka Yorumu:\n" + llm_yorum + "\n\n⚠️ Bu uygulama yalnızca araştırma amaçlıdır."

    # Sonucu kaydet (indirme için)
   
    save_path = os.path.join(tempfile.gettempdir(), "sonuc_overlay.png")
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return overlay, mask_rgb, bilgi, round(guven, 1)


# ── Çoklu görüntü analizi ──────────────────────────
def analyze_multiple(files):
    if not files:
        return "Lütfen görüntü yükleyin."

    results = []
    for file in files:
        image_bgr = cv2.imread(file.name)
        if image_bgr is None:
            results.append(f"📁 {os.path.basename(file.name)}\n   ⚠️ Görüntü okunamadı")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        prob, pred_binary = predict_single(image_rgb)
        _, _, guven = make_visuals(image_rgb, prob, pred_binary)
        lezyon_oran = (pred_binary.sum() / pred_binary.size) * 100
        tespit = "✅ Lezyon var" if lezyon_oran > 1 else "❌ Lezyon yok"
        results.append(
            f"📁 {os.path.basename(file.name)}\n"
            f"   {tespit} | Alan: %{lezyon_oran:.1f} | Güven: %{guven:.1f}"
        )

    return "\n\n".join(results)


# ── Arayüz ─────────────────────────────────────────
with gr.Blocks(title="Deri Lezyonu Bölütleme", theme=themes.Soft()) as demo:

    gr.Markdown("""
    # 🔬 Derin Öğrenme Tabanlı Deri Lezyonu Bölütleme
    **Attention U-Net (ResNet50 Encoder)** | ISIC 2017-2018 Veri Seti
    """)

    with gr.Tabs():

        # ── Tab 1: Tek Görüntü ──────────────────────
        with gr.Tab("🖼️ Görüntü Analizi"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Deri Görüntüsü Yükle", type="numpy", height=320)
                    btn_single  = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

                with gr.Column():
                    output_overlay = gr.Image(label="Lezyon Tespiti (Yeşil=Lezyon, Kırmızı=Sınır)", height=320)

            with gr.Row():
                with gr.Column():
                    output_mask = gr.Image(label="Bölütleme Maskesi", height=250)
                with gr.Column():
                    output_text  = gr.Textbox(label="Analiz Sonucu", lines=7)
                    guven_bar    = gr.Slider(label="Güven Skoru (%)", minimum=0, maximum=100,
                                            interactive=False, value=0)
                    

            btn_single.click(
                fn      = analyze_single,
                inputs  = [input_image],
                outputs = [output_overlay, output_mask, output_text, guven_bar],
            )

           

        

    # Örnek görüntüler
    example_dir = os.path.join(os.path.dirname(__file__), "..", "data", "images")
    example_files = []
    if os.path.exists(example_dir):
        for f in sorted(os.listdir(example_dir))[:3]:
            full_path = os.path.join(example_dir, f)
            if f.endswith(".jpg") and os.path.exists(full_path):
                example_files.append([full_path])
    if example_files:
        gr.Examples(examples=example_files, inputs=input_image, label="Örnek Görüntüler")


if __name__ == "__main__":
    demo.launch(share=False)