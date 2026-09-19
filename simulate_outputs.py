"""
simulate_outputs.py
====================
Membuat outputs/results.json dan outputs/forecast_data.csv dari
data/traffic_data.csv TANPA menjalankan ARIMA/BiLSTM sungguhan.

Gunakan script ini HANYA untuk demonstrasi tampilan dashboard secara
end-to-end sebelum training sesungguhnya selesai. Untuk hasil
penelitian sesungguhnya, gunakan Training_ARIMA_BiLSTM.ipynb di
Google Colab — script tersebut akan menghasilkan output dengan
format yang sama persis.

Cara pakai:
    python generate_dummy_data.py   # jika belum punya data/traffic_data.csv
    python simulate_outputs.py
    streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd

np.random.seed(123)

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/traffic_data.csv", parse_dates=["Date"])
df["Visits"] = df["Visits"].interpolate(method="linear", limit_direction="both")

ts = df.set_index("Date")["Visits"].astype(float)

split_idx = int(len(ts) * 0.8)
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

# Simulasi prediksi ARIMA: noise lebih besar
arima_noise = np.random.normal(0, test.std() * 0.18, len(test))
arima_pred = test.values * 0.97 + arima_noise + test.mean() * 0.02

# Simulasi prediksi BiLSTM: noise lebih kecil (menangkap pola non-linier lebih baik)
bilstm_noise = np.random.normal(0, test.std() * 0.11, len(test))
bilstm_pred = test.values * 0.995 + bilstm_noise


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


arima_metrics = compute_metrics(test.values, arima_pred)
bilstm_metrics = compute_metrics(test.values, bilstm_pred)

# ── forecast_data.csv ────────────────────────────────────────────────────────
full_result = pd.DataFrame({"Date": ts.index, "Actual": ts.values})
full_result["ARIMA_Pred"] = np.nan
full_result["BiLSTM_Pred"] = np.nan
full_result.loc[split_idx:, "ARIMA_Pred"] = arima_pred
full_result.loc[split_idx:, "BiLSTM_Pred"] = bilstm_pred
full_result["DataSplit"] = ["Train"] * len(train) + ["Test"] * len(test)
full_result.to_csv("outputs/forecast_data.csv", index=False)

# ── results.json ──────────────────────────────────────────────────────────────
best_model = "ARIMA" if arima_metrics["RMSE"] < bilstm_metrics["RMSE"] else "BiLSTM"

results = {
    "page_name": "Portal_Berita_Online_id.wikipedia.org (SIMULASI)",
    "n_total": int(len(ts)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "arima": {
        "order": [2, 1, 2],
        "metrics": arima_metrics,
        "ljung_box_pvalue": 0.421,
    },
    "bilstm": {
        "units": 64,
        "lookback": 14,
        "dropout": 0.2,
        "batch_size": 16,
        "epochs_run": 47,
        "metrics": bilstm_metrics,
    },
    "best_model": best_model,
    "adf_pvalue": 0.012,
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Tersimpan: outputs/forecast_data.csv")
print("Tersimpan: outputs/results.json")
print(json.dumps(results, indent=2))
