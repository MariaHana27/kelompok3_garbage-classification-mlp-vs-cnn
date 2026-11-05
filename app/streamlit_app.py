"""Aplikasi demo pemilah sampah otomatis (Streamlit).

Fitur unik:
1. Grad-CAM            : area gambar yang paling memengaruhi prediksi.
2. Confidence threshold: prediksi yang kurang meyakinkan ditolak.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Pemilah Sampah Otomatis", page_icon="♻️", layout="wide")

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "models", HERE / "models", HERE]
IMG_SIZE = (224, 224)

KATEGORI = {
    "battery": "B3 (limbah berbahaya) - serahkan ke titik pengumpulan baterai bekas",
    "biological": "Organik - bisa dikomposkan",
    "brown-glass": "Anorganik (kaca) - daur ulang kaca",
    "green-glass": "Anorganik (kaca) - daur ulang kaca",
    "white-glass": "Anorganik (kaca) - daur ulang kaca",
    "cardboard": "Anorganik (kertas/karton) - daur ulang",
    "paper": "Anorganik (kertas) - daur ulang",
    "clothes": "Tekstil - donasikan atau daur ulang",
    "metal": "Anorganik (logam) - daur ulang",
    "plastic": "Anorganik (plastik) - daur ulang",
    "shoes": "Tekstil/karet - donasikan atau daur ulang",
    "trash": "Residu - buang ke tempat sampah umum",
}


def _find(name, required=True):
    for d in CANDIDATES:
        if (d / name).exists():
            return d / name
    if required:
        raise FileNotFoundError(f"{name} tidak ditemukan di folder models/.")
    return None


@st.cache_resource(show_spinner="Memuat model...")
def load_assets():
    model = tf.keras.models.load_model(_find("best_model.keras"), compile=False)
    class_names = json.load(open(_find("class_names.json")))
    cfg = _find("config.json", required=False)
    threshold = json.load(open(cfg)).get("threshold", 0.7) if cfg else 0.7
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    gap = next((l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)), None)
    head = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][-1]
    return model, class_names, threshold, base, gap, head


MODEL, CLASS_NAMES, THRESHOLD, BASE, GAP, HEAD = load_assets()


def gradcam_overlay(arr, class_idx):
    if BASE is None or GAP is None:
        return None
    x = tf.convert_to_tensor(arr[None], dtype=tf.float32) / 127.5 - 1.0
    with tf.GradientTape() as tape:
        tape.watch(x)
        fmap = BASE(x, training=False)
        tape.watch(fmap)
        preds = HEAD(GAP(fmap))
        score = preds[:, class_idx]
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam_img = np.array(Image.fromarray(np.uint8(255 * cam)).resize(IMG_SIZE, Image.BILINEAR)) / 255.0
    heat = plt.get_cmap("jet")(cam_img)[..., :3]
    out = np.clip(0.55 * arr / 255.0 + 0.45 * heat, 0, 1)
    return (out * 255).astype("uint8")


def predict(image: Image.Image):
    # resize identik dengan pipeline pelatihan (tf.image.resize, bilinear)
    arr = tf.image.resize(np.array(image.convert("RGB"), dtype="float32"), IMG_SIZE).numpy()
    probs = MODEL.predict(arr[None], verbose=0)[0]
    return arr, probs


st.title("♻️ Pemilah Sampah Rumah Tangga Otomatis")
st.caption("Unggah foto sampah. Sistem memprediksi jenisnya (12 kelas), menampilkan Grad-CAM, "
           "dan menolak prediksi yang kurang meyakinkan.")

file = st.file_uploader("Unggah foto sampah", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file)
    arr, probs = predict(image)
    idx = int(probs.argmax())
    top, conf = CLASS_NAMES[idx], float(probs[idx])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Foto asli")
        st.image(image, use_container_width=True)
    with c2:
        st.subheader("Grad-CAM")
        cam = gradcam_overlay(arr, idx)
        if cam is not None:
            st.image(cam, caption="Area yang menjadi dasar prediksi", use_container_width=True)
        else:
            st.info("Grad-CAM tidak tersedia untuk model ini.")

    if conf < THRESHOLD:
        st.warning(f"Model tidak yakin (confidence {conf:.1%} < threshold {THRESHOLD:.0%}). "
                   f"Dugaan terdekat: **{top}**. Coba foto lain yang lebih jelas atau pastikan "
                   f"objek adalah sampah rumah tangga.")
    else:
        st.success(f"Prediksi: **{top}** ({conf:.1%})")
        st.write(f"Kategori: {KATEGORI.get(top, '-')}")

    st.subheader("Top-3 prediksi")
    for i in probs.argsort()[::-1][:3]:
        st.write(f"{CLASS_NAMES[i]}: {probs[i]:.1%}")
        st.progress(float(probs[i]))
else:
    st.info("Belum ada gambar. Unggah foto sampah untuk memulai.")
