# 🧠 Klasifikasi Citra Intel dengan CNN + Transfer Learning

Proyek ini melakukan klasifikasi gambar dari **dataset Intel Image Classification** menggunakan **Transfer Learning (MobileNetV2)** yang dikombinasikan dengan layer tambahan **Conv2D dan Pooling**. Model dibangun dengan TensorFlow/Keras dan diekspor dalam berbagai format untuk keperluan deployment.

Model dikembangkan untuk mengklasifikasikan gambar menjadi 6 kategori:
- `buildings` (bangunan)
- `forest` (hutan)
- `glacier` (gletser)
- `mountain` (gunung)
- `sea` (laut)
- `street` (jalan)

---

## 📁 Struktur Proyek

```bash
submission/
├── saved_model/              → Format SavedModel TensorFlow
├── tflite/
│   ├── model.tflite          → Model TFLite untuk mobile/embedded
│   └── label.txt             → Daftar label kelas
├── tfjs_model/
│   ├── model.json            → Model TensorFlow.js untuk browser
│   └── group1-shard1of1.bin  → Bobot model dalam format binari
├── notebook.ipynb            → Notebook pelatihan, evaluasi, dan inferensi
├── requirements.txt          → Daftar dependensi Python
└── README.md                 → Dokumen deskripsi proyek
```

---

## 🧠 Arsitektur Model

- **Pretrained Base**: MobileNetV2 (`include_top=False`, bobot dari ImageNet)
- **Tambahan Layer Kustom**:
  - Conv2D(32) + MaxPooling2D
  - GlobalAveragePooling2D
  - Dropout(0.5)
  - Dense(128) + Dense(6, softmax)
- **Loss Function**: categorical_crossentropy
- **Optimizer**: Adam

Struktur ini dirancang agar memenuhi ketentuan penggunaan CNN dan mampu mencapai akurasi tinggi.

---

## 📊 Hasil Pelatihan

| Metrik               | Nilai   |
|----------------------|---------|
| Akurasi Training     | >91%    |
| Akurasi Validasi     | >91%    |
| Akurasi Pengujian    | >91%    |

Model mampu melakukan generalisasi dengan baik pada data yang belum pernah dilihat.

---

## 🚀 Cara Menjalankan Proyek

### 📊 Training & Evaluasi (Jupyter Notebook)
1. Instal semua dependensi:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. Jalankan notebook:
   \`\`\`bash
   jupyter notebook notebook.ipynb
   \`\`\`

3. Ikuti urutan sel untuk melatih, menguji, dan melakukan prediksi pada gambar.

### 🌐 Web Application (Streamlit)

#### 🏠 Local Development
1. Instal dependensi lengkap:
   \`\`\`bash
   pip install -r requirements_full.txt
   \`\`\`

2. Jalankan aplikasi web:
   \`\`\`bash
   # Opsi 1: Menggunakan script otomatis
   python3 run_app.py
   
   # Opsi 2: Langsung dengan streamlit
   streamlit run app.py
   \`\`\`

3. Buka browser di `http://localhost:8501`

#### ☁️ Live Demo
🚀 **Akses aplikasi web secara langsung:** [classifit.streamlit.app](https://classifit.streamlit.app)

#### 📋 Features
- 📤 Upload gambar (PNG, JPG, JPEG)
- 🎯 Prediksi real-time dengan confidence score
- 📊 Top 3 prediksi dengan visualisasi progress bar
- 🎨 Interface modern dengan dark/light mode support
- 📱 Responsive design untuk desktop dan mobile
- 🌓 Auto-adaptive styling untuk berbagai theme

---

## 📦 Format Model Output

| Format         | Lokasi                  | Keterangan                       |
|----------------|--------------------------|-----------------------------------|
| SavedModel     | `saved_model/`           | Untuk deployment server/serving  |
| TensorFlow Lite| `tflite/model.tflite`    | Untuk mobile/embedded device     |
| TFJS           | `tfjs_model/model.json`  | Untuk aplikasi berbasis browser  |

---

Dokumen ini disusun sebagai bagian dari tugas Proyek Klasifikasi Gambar