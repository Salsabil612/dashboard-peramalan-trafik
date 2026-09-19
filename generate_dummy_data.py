"""
generate_dummy_data.py
========================
Generator dataset dummy trafik harian website berita online,
mensimulasikan karakteristik dataset Web Traffic Time Series
Forecasting (Kaggle/Google, 2017): tren, musiman mingguan,
event/hari libur, dan noise acak.

Output: data/traffic_data.csv (format long: Date, Visits)

Cara pakai:
    python generate_dummy_data.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Konfigurasi ──────────────────────────────────────────────────────────────
N_DAYS = 550  # sesuai batasan masalah: ±550 hari
START_DATE = datetime(2023, 1, 1)
PAGE_NAME = "Portal_Berita_Online_id.wikipedia.org"

os.makedirs("data", exist_ok=True)

# ── Komponen Time Series ─────────────────────────────────────────────────────
dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]
t = np.arange(N_DAYS)

# 1. Tren (trend) - pertumbuhan bertahap dengan sedikit fluktuasi
trend = 8000 + 6 * t + 500 * np.sin(t / 90)

# 2. Musiman mingguan (weekly seasonality) - trafik lebih tinggi di hari kerja
weekday = np.array([d.weekday() for d in dates])  # 0=Senin ... 6=Minggu
weekly_pattern = np.array([1.15, 1.20, 1.18, 1.15, 1.10, 0.75, 0.70])  # Sen-Min
seasonal = trend * (weekly_pattern[weekday] - 1)

# 3. Event / hari libur nasional - lonjakan trafik signifikan
event_days = [10, 45, 89, 120, 180, 210, 250, 300, 333, 365, 400, 450, 480, 520]
event_spike = np.zeros(N_DAYS)
for ed in event_days:
    if ed < N_DAYS:
        for offset, mag in zip([-1, 0, 1, 2], [0.3, 1.0, 0.6, 0.25]):
            idx = ed + offset
            if 0 <= idx < N_DAYS:
                event_spike[idx] += trend[idx] * mag * np.random.uniform(0.6, 1.0)

# 4. Residual / noise acak
noise = np.random.normal(0, trend * 0.05, N_DAYS)

# Gabungkan semua komponen
visits = trend + seasonal + event_spike + noise
visits = np.maximum(visits, 0).round().astype(int)

# 5. Simulasikan beberapa missing values (sesuai karakteristik dataset asli)
missing_idx = np.random.choice(N_DAYS, size=8, replace=False)
visits_with_na = visits.astype(float)
visits_with_na[missing_idx] = np.nan

# ── Buat DataFrame (format long) ─────────────────────────────────────────────
df = pd.DataFrame({
    "Page": PAGE_NAME,
    "Date": [d.strftime("%Y-%m-%d") for d in dates],
    "Visits": visits_with_na
})

df.to_csv("data/traffic_data.csv", index=False)

print("Dataset dummy berhasil dibuat: data/traffic_data.csv")
print(f"Jumlah baris: {len(df)}")
print(f"Periode: {df['Date'].min()} s.d. {df['Date'].max()}")
print(f"Missing values: {df['Visits'].isna().sum()}")
print(f"Statistik Visits:\n{df['Visits'].describe()}")
