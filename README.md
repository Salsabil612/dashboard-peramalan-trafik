# Dashboard Peramalan Trafik Website Berita Online
### Analisis Perbandingan ARIMA dan BiLSTM Berbasis Dashboard

Skripsi — Salsabil Zahra (4123049), Program Studi Manajemen Informatika,
STMIK Al Muslim Bekasi.

---

## Struktur Folder

```
app/
├── app.py                          # Dashboard Streamlit (jalankan ini)
├── requirements.txt                # Library untuk dashboard
├── requirements_training.txt       # Library untuk notebook training (Colab)
├── generate_dummy_data.py          # Generator dataset dummy (opsional)
├── Training_ARIMA_BiLSTM.ipynb     # Notebook training (jalankan di Google Colab)
├── data/
│   └── traffic_data.csv            # Dataset dummy (hasil generate_dummy_data.py)
└── outputs/                         # Letakkan hasil training Colab di sini
    ├── results.json
    ├── forecast_data.csv
    ├── bilstm_model.h5
    └── scaler.save
```

---

## Cara Menjalankan (Alur Lengkap)

### 1. Siapkan Dataset

Gunakan CSV trafik website kamu sendiri, dengan salah satu format:

**Format A — Long (direkomendasikan):**

| Date       | Visits |
|------------|--------|
| 2023-01-01 | 9210   |
| 2023-01-02 | 9540   |
| ...        | ...    |

**Format B — Wide ala Kaggle (Web Traffic Time Series Forecasting):**

| Page                  | 2023-01-01 | 2023-01-02 | ... |
|-----------------------|------------|------------|-----|
| Halaman_A.wikipedia   | 9210       | 9540       | ... |
| Halaman_B.wikipedia   | 1200       | 1340       | ... |

Jika tidak punya data sendiri, jalankan `python generate_dummy_data.py`
untuk membuat dataset dummy (`data/traffic_data.csv`) yang mensimulasikan
tren, musiman mingguan, dan event/hari libur sesuai karakteristik
dataset Kaggle.

### 2. Training Model (Google Colab)

1. Buka [Google Colab](https://colab.research.google.com/), buat notebook baru.
2. Upload `Training_ARIMA_BiLSTM.ipynb`.
3. Jalankan semua cell secara berurutan (Runtime → Run all).
   - Saat diminta, upload file CSV dataset kamu.
   - Notebook otomatis mendeteksi format CSV (long/wide), melakukan
     preprocessing, training ARIMA (Box-Jenkins + auto_arima) dan
     BiLSTM (TensorFlow/Keras), serta menghitung RMSE, MAE, MAPE.
4. Di akhir notebook, 4 file akan otomatis terdownload:
   - `forecast_data.csv` — data historis + hasil prediksi
   - `results.json` — metrik evaluasi & parameter model
   - `bilstm_model.h5` — model BiLSTM terlatih
   - `scaler.save` — MinMaxScaler yang digunakan

### 3. Jalankan Dashboard

1. Letakkan keempat file dari Colab ke folder `outputs/` pada project ini.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan dashboard:
   ```bash
   streamlit run app.py
   ```
4. Dashboard akan terbuka di browser (default: `http://localhost:8501`).

> **Catatan:** Jika folder `outputs/` belum berisi hasil training, dashboard
> akan menampilkan **data demo** sebagai contoh tampilan, sehingga kamu bisa
> menguji UI sebelum proses training selesai.

---

## Fitur Dashboard

1. **📈 Data Historis** — visualisasi trafik harian + statistik ringkas
   (rata-rata, maksimum, minimum). Bisa upload CSV baru untuk melihat
   data lain.
2. **🔀 Perbandingan Prediksi** — grafik aktual vs prediksi ARIMA vs
   BiLSTM (seluruh data atau hanya data uji), serta grafik residual.
3. **📋 Evaluasi Model** — tabel dan grafik batang perbandingan RMSE,
   MAE, MAPE; detail parameter ARIMA (orde p,d,q, uji ADF, Ljung-Box)
   dan BiLSTM (units, lookback, dropout, epoch).
4. **🏆 Rekomendasi** — kesimpulan otomatis model terbaik beserta
   rekomendasi praktis untuk manajemen kapasitas server.

---

## Pengujian Dashboard (Black Box Testing)

Sesuai Bab III, pengujian dashboard mencakup:

- [ ] Grafik data historis dimuat dan dirender dengan benar
- [ ] Tombol/kontrol (radio button, file uploader, tab) merespons input
- [ ] Tabel metrik evaluasi menampilkan nilai yang akurat
- [ ] Grafik perbandingan prediksi tampil bersamaan dan dapat dibaca
- [ ] Dashboard dapat diakses di berbagai browser (Chrome, Firefox, Edge)
