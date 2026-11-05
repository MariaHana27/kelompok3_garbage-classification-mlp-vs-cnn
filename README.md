# Perbandingan MLP dan CNN untuk Pemilahan Sampah Rumah Tangga Otomatis Berbasis Citra

Proyek UTS Pembelajaran Mendalam (Kelompok 3). Membandingkan tiga model untuk mengklasifikasikan 12 jenis sampah dari citra, lalu men-deploy model terbaik sebagai aplikasi web.

## Tautan

| Item | Link |
|---|---|
| Aplikasi yang sudah di-deploy (Streamlit) | https://kelompok3garbage-classification-mlp-vs-cnn-7kr4y4gklgqfzvbyhxw.streamlit.app/ |
| Video demo dan tahapan deployment | ISI_LINK_VIDEO |

## Dataset

[Garbage Classification (Kaggle)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification): 15.515 citra, 12 kelas (`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`). Dataset tidak disertakan di repository ini; notebook mengunduhnya otomatis lewat `kagglehub`.

## Hasil (data uji, 2.325 citra)

| Model | Parameter | Waktu latih (detik) | Test Accuracy | Macro F1 |
|---|---|---|---|---|
| MLP | 6.429.452 | 579,7 | 55,70% | 0,4769 |
| CNN dari nol | 458.732 | 1.634,0 | 75,48% | 0,7210 |
| **MobileNetV2 (transfer learning)** | 2.273.356 | 774,2 | **96,00%** | **0,9466** |

Model terbaik: MobileNetV2. Dengan fitur confidence threshold 0,8, akurasi pada prediksi yang diterima naik menjadi 98,22% (94,41% data diterima).

## Arsitektur dan Fitur

- **MLP (baseline):** Resize 64x64, Flatten, Dense 512, Dense 256, Softmax (ReLU, BatchNorm, Dropout 0,4).
- **CNN dari nol:** 4 blok Conv-BatchNorm-ReLU-MaxPool (32/64/128/256 filter), GlobalAveragePooling, Dense 256, Softmax (Dropout 0,5).
- **MobileNetV2:** bobot ImageNet; fase 1 melatih kepala klasifikasi, fase 2 fine-tuning 30 layer terakhir.
- **Fitur unik 1, Grad-CAM:** peta panas area citra yang menjadi dasar prediksi.
- **Fitur unik 2, confidence threshold:** prediksi dengan confidence di bawah 0,8 ditolak.

## Struktur Repository

```
kelompok3_garbage-classification-mlp-vs-cnn/
├── README.md
├── requirements.txt
├── notebooks/
│   └── garbage_classification_mlp_vs_cnn.ipynb
├── app/
│   └── streamlit_app.py
├── models/
│   ├── best_model.keras
│   ├── class_names.json
│   └── config.json
└── assets/
    └── grafik (.png), tabel hasil (.csv), history.json, env_info.json, classification_report.txt
```

## Cara Menjalankan

**1. Notebook (Google Colab, GPU Tesla T4)**
1. Buka `notebooks/garbage_classification_mlp_vs_cnn.ipynb` di Google Colab.
2. Aktifkan GPU: Runtime, Change runtime type, GPU.
3. Jalankan semua sel dari atas ke bawah. Hasilnya tersimpan di folder `models/` dan `assets/`.

**2. Aplikasi web di komputer sendiri**
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
Buka http://localhost:8501.

**3. Deploy ke Streamlit Community Cloud**
1. Buka https://share.streamlit.io dan masuk dengan akun GitHub.
2. Klik Create app, pilih repository ini dan branch `main`.
3. Isi Main file path dengan `app/streamlit_app.py`.
4. Di Advanced settings pilih Python 3.12, lalu klik Deploy.

## Anggota Kelompok

| NIM | Nama | Tanggung Jawab |
|---|---|---|
| 231112060 | Steven Raffael Antonius Sidauruk | Dataset dan EDA; kompilasi laporan |
| 231112393 | William Patar Wijaya Marpaung | Preprocessing; pengecekan tautan |
| 231112124 | Maria Hana Gultom | Arsitektur CNN dan Transfer Learning; GitHub dan README |
| 231112294 | Arisanti Nurlaili | Arsitektur MLP; eksperimen hyperparameter |
| 231112002 | Muhammad Aswan Tagiy | Deployment, demo, dan video |
| 231112256 | Daniel Pangihutan Siregar | Evaluasi dan analisis hasil |
