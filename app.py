"""
app.py
======
Dashboard Interaktif — Analisis Perbandingan ARIMA dan BiLSTM untuk
Peramalan Trafik Website Berita Online

Skripsi: Salsabil Zahra (4123049) — STMIK Al Muslim Bekasi

Jalankan dengan:
    streamlit run app.py

Mode data:
1. Default    -> membaca hasil training dari Google Colab (outputs/results.json
                 + outputs/forecast_data.csv)
2. Upload CSV -> upload data trafik baru, lalu (opsional) latih ulang ARIMA &
                 BiLSTM LANGSUNG di dashboard ini (hanya disarankan saat
                 dijalankan lokal, karena training BiLSTM cukup berat).
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # kurangi log TensorFlow yang berisik

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Library untuk retraining langsung di dashboard — opsional, dibungkus try/except
# supaya dashboard tetap bisa jalan (mode visualisasi saja) walau library ini
# belum terpasang.
try:
    import pmdarima as pm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TRAINING_LIBS_AVAILABLE = True
except ImportError:
    TRAINING_LIBS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Peramalan Trafik Website",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = "outputs"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")
FORECAST_CSV = os.path.join(OUTPUT_DIR, "forecast_data.csv")
MIN_ROWS_FOR_TRAINING = 60  # minimal jumlah hari data agar training di dashboard layak dilakukan

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        color: #0D1B40;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #028090;
        font-weight: 600;
        margin-top: -8px;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #028090;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #64748B; }
    .recommendation-box {
        background-color: #F0FDF9;
        border: 1px solid #02C39A;
        border-left: 6px solid #02C39A;
        border-radius: 8px;
        padding: 18px 20px;
        margin-top: 10px;
    }
    .info-box {
        background-color: #F0F4F8;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #64748B;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — UMUM
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Hitung RMSE, MAE, MAPE, SMAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "SMAPE": np.nan}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, np.nan, denom)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "SMAPE": smape}


def get_metric(metrics_dict, key):
    """Ambil nilai metrik dengan aman; fallback np.nan jika key belum ada
    (mis. results.json lama yang dibuat sebelum SMAPE ditambahkan)."""
    return metrics_dict.get(key, np.nan)


def load_as_long_format(df):
    """Deteksi format CSV (long atau wide ala Kaggle) -> konversi ke (Date, Visits)."""
    cols_lower = [c.lower() for c in df.columns]

    if "date" in cols_lower:
        date_col = df.columns[cols_lower.index("date")]
        value_candidates = [c for c in df.columns if c != date_col]
        value_col = None
        for c in value_candidates:
            if c.lower() in ("visits", "visit", "value", "traffic"):
                value_col = c
                break
        if value_col is None:
            numeric_cols = df[value_candidates].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("Tidak ditemukan kolom numerik untuk nilai trafik.")
            value_col = numeric_cols[0]
        long_df = df[[date_col, value_col]].copy()
        long_df.columns = ["Date", "Visits"]
        page_name = "Data Upload (Single Series)"

    elif "page" in cols_lower:
        page_col = df.columns[cols_lower.index("page")]
        date_cols = [c for c in df.columns if c != page_col]
        variances = df[date_cols].astype(float).std(axis=1, skipna=True)
        best_idx = variances.idxmax()
        page_name = df.loc[best_idx, page_col]
        series = df.loc[best_idx, date_cols].astype(float)
        long_df = pd.DataFrame({"Date": date_cols, "Visits": series.values})

    else:
        raise ValueError('Format CSV tidak dikenali. Pastikan ada kolom "Date" atau "Page".')

    long_df["Date"] = pd.to_datetime(long_df["Date"])
    long_df = long_df.sort_values("Date").reset_index(drop=True)
    long_df["Visits"] = long_df["Visits"].interpolate(method="linear", limit_direction="both")
    return long_df, page_name


@st.cache_data
def load_default_results():
    """Load hasil training dari notebook (results.json + forecast_data.csv)."""
    if os.path.exists(RESULTS_JSON) and os.path.exists(FORECAST_CSV):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        forecast_df = pd.read_csv(FORECAST_CSV, parse_dates=["Date"])
        return results, forecast_df
    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION — RETRAIN LANGSUNG DI DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

def train_models_on_upload(df, page_name, fast_mode=True):
    """
    Melatih ulang ARIMA (auto_arima) dan BiLSTM (TensorFlow/Keras) langsung
    di dashboard, menggunakan data upload (kolom Date, Visits).

    Mengikuti tahapan yang sama seperti notebook Training_ARIMA_BiLSTM.ipynb:
    split 80/20 berurutan, ADF test, auto_arima (AIC), Ljung-Box, BiLSTM dengan
    Early Stopping. `fast_mode` mengecilkan ruang pencarian ARIMA dan jumlah
    epoch BiLSTM supaya waktu tunggu di laptop tidak terlalu lama.

    Mengembalikan (results_dict, forecast_df) dengan skema yang sama persis
    dengan hasil training di Colab, sehingga bisa langsung dipakai oleh
    tab Perbandingan Prediksi, Evaluasi Model, dan Rekomendasi.
    """
    if not TRAINING_LIBS_AVAILABLE:
        raise RuntimeError(
            "Library training belum terpasang. Jalankan: "
            "pip install pmdarima tensorflow scikit-learn statsmodels"
        )

    df = df.sort_values("Date").reset_index(drop=True)
    n_total = len(df)
    if n_total < MIN_ROWS_FOR_TRAINING:
        raise ValueError(
            f"Data terlalu pendek untuk dilatih (minimal {MIN_ROWS_FOR_TRAINING} hari, "
            f"data Anda hanya {n_total} hari)."
        )

    split_idx = int(n_total * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    n_train, n_test = len(train_df), len(test_df)

    # 1) Uji stasioneritas ADF pada data latih
    adf_pvalue = float(adfuller(train_df["Visits"])[1])

    # 2) ARIMA — identifikasi & estimasi otomatis (kriteria AIC)
    max_pq = 3 if fast_mode else 5
    arima_model = pm.auto_arima(
        train_df["Visits"],
        start_p=0, start_q=0,
        max_p=max_pq, max_q=max_pq,
        d=None,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    arima_order = arima_model.order

    residuals = arima_model.resid()
    ljung_box_pvalue = float(acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].iloc[0])

    arima_forecast = np.asarray(arima_model.predict(n_periods=n_test))

    # 3) BiLSTM — normalisasi, sequence, arsitektur, training dengan Early Stopping
    LOOKBACK = 14
    BILSTM_UNITS = 32 if fast_mode else 64
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 16
    MAX_EPOCHS = 30 if fast_mode else 100
    PATIENCE = 5 if fast_mode else 10

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[["Visits"]])
    scaled_all = scaler.transform(df[["Visits"]]).flatten()
    scaled_train = scaled_all[:n_train]
    scaled_test_input = scaled_all[n_train - LOOKBACK:]

    def make_sequences(series, lookback):
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i - lookback:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    X_train_full, y_train_full = make_sequences(scaled_train, LOOKBACK)
    X_test, _ = make_sequences(scaled_test_input, LOOKBACK)

    if len(X_train_full) < 10 or len(X_test) == 0:
        raise ValueError(
            "Data latih terlalu pendek untuk membentuk sequence BiLSTM (lookback=14 hari). "
            "Tambahkan lebih banyak data historis."
        )

    val_split = max(1, int(len(X_train_full) * 0.9))
    X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
    X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]
    if len(X_val) == 0:
        X_val, y_val = X_train[-1:], y_train[-1:]  # jaga-jaga data sangat pendek

    X_train = X_train.reshape(-1, LOOKBACK, 1)
    X_val = X_val.reshape(-1, LOOKBACK, 1)
    X_test = X_test.reshape(-1, LOOKBACK, 1)

    tf.random.set_seed(42)
    bilstm_model = Sequential([
        Bidirectional(LSTM(BILSTM_UNITS), input_shape=(LOOKBACK, 1)),
        Dropout(DROPOUT_RATE),
        Dense(1),
    ])
    bilstm_model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = bilstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    epochs_run = len(history.history["loss"])

    bilstm_pred_scaled = bilstm_model.predict(X_test, verbose=0).flatten()
    bilstm_forecast = scaler.inverse_transform(bilstm_pred_scaled.reshape(-1, 1)).flatten()

    # 4) Evaluasi
    actual_test = test_df["Visits"].values
    arima_metrics = compute_metrics(actual_test, arima_forecast)
    bilstm_metrics = compute_metrics(actual_test, bilstm_forecast)
    best_model = "ARIMA" if arima_metrics["RMSE"] < bilstm_metrics["RMSE"] else "BiLSTM"

    # 5) Susun forecast_df & results (skema sama persis dengan output Colab)
    forecast_df = pd.DataFrame({"Date": df["Date"], "Actual": df["Visits"]})
    forecast_df["ARIMA_Pred"] = np.nan
    forecast_df["BiLSTM_Pred"] = np.nan
    forecast_df.loc[n_train:, "ARIMA_Pred"] = arima_forecast
    forecast_df.loc[n_train:, "BiLSTM_Pred"] = bilstm_forecast
    forecast_df["DataSplit"] = ["Train"] * n_train + ["Test"] * n_test

    results = {
        "page_name": str(page_name),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "arima": {
            "order": list(arima_order),
            "metrics": arima_metrics,
            "ljung_box_pvalue": round(ljung_box_pvalue, 4),
        },
        "bilstm": {
            "units": BILSTM_UNITS,
            "lookback": LOOKBACK,
            "dropout": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "epochs_run": int(epochs_run),
            "metrics": bilstm_metrics,
        },
        "best_model": best_model,
        "adf_pvalue": round(adf_pvalue, 4),
    }
    return results, forecast_df


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file = None
retrain_clicked = False
fast_mode = True

with st.sidebar:
    st.markdown("### 📊 Dashboard Peramalan Trafik")
    st.markdown("**Skripsi:** Analisis Perbandingan ARIMA dan BiLSTM untuk Peramalan Trafik Website Berita Online")
    st.markdown("**Penyusun:** Salsabil Zahra (4123049)")
    st.markdown("**Prodi:** Manajemen Informatika — STMIK Al Muslim")
    st.divider()

    st.markdown("#### 📁 Sumber Data")
    data_source = st.radio(
        "Pilih sumber data:",
        ["Gunakan hasil training (outputs/)", "Upload CSV trafik baru"],
        index=0,
    )

    if data_source == "Upload CSV trafik baru":
        uploaded_file = st.file_uploader(
            "Upload file CSV",
            type=["csv"],
            help="Format long: kolom 'Date' dan 'Visits'. Format wide ala Kaggle: kolom 'Page' + kolom-kolom tanggal."
        )
        st.caption(
            "ℹ️ Upload CSV akan menampilkan **visualisasi data historis**. "
            "Secara default, prediksi ARIMA & BiLSTM tetap berasal dari model yang "
            "sudah dilatih di Google Colab. Jika ingin, Anda bisa **melatih ulang "
            "kedua model langsung memakai data ini** lewat tombol di bawah."
        )

        st.write("")
        if TRAINING_LIBS_AVAILABLE:
            st.markdown("#### 🔄 Latih Ulang Model")
            fast_mode = st.checkbox(
                "Mode cepat (disarankan)",
                value=True,
                help="Mengecilkan ruang pencarian ARIMA (max p,q = 3) dan epoch BiLSTM "
                     "(maks. 30 epoch) agar training lebih cepat di laptop. Matikan untuk "
                     "hasil setara pengaturan di notebook Colab (lebih lama)."
            )
            retrain_clicked = st.button("🚀 Latih Ulang dengan Data Ini", use_container_width=True)
            st.caption(
                "⏱️ Training berjalan langsung di komputer Anda — bisa memakan waktu "
                "sekitar 30 detik s.d. beberapa menit tergantung panjang data & spesifikasi laptop."
            )
        else:
            st.warning(
                "⚠️ Fitur 'Latih Ulang' belum aktif karena library training belum terpasang. "
                "Jalankan `pip install pmdarima tensorflow scikit-learn statsmodels` "
                "lalu restart aplikasi untuk mengaktifkannya."
            )

        if st.session_state.get("retrained_results") is not None:
            st.divider()
            st.success(f"✅ Menampilkan hasil retraining: **{st.session_state.get('retrained_page_name', '')}**")
            if st.button("↩️ Reset ke Hasil Training Awal (Colab)", use_container_width=True):
                st.session_state.pop("retrained_results", None)
                st.session_state.pop("retrained_forecast_df", None)
                st.session_state.pop("retrained_page_name", None)
                st.rerun()

    st.divider()
    st.markdown("#### ⚙️ Tentang Model")
    st.markdown(
        "- **ARIMA**: Box-Jenkins + `auto_arima` (AIC)\n"
        "- **BiLSTM**: TensorFlow/Keras, Dropout, Early Stopping\n"
        "- **Training default**: dilakukan offline di Google Colab\n"
        "- **Training opsional**: langsung di dashboard (data upload)\n"
        "- **Metrik**: RMSE, MAE, MAPE, SMAPE"
    )


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Dashboard Peramalan Trafik Website Berita Online</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis Perbandingan ARIMA dan BiLSTM Berbasis Dashboard</p>', unsafe_allow_html=True)
st.write("")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
results, forecast_df = load_default_results()
user_df = None
user_page_name = None

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        user_df, user_page_name = load_as_long_format(raw_df)
        st.success(f"✅ Data berhasil dimuat: **{user_page_name}** ({len(user_df)} baris)")
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")

# ── Jalankan retraining jika tombol ditekan ────────────────────────────────
if retrain_clicked:
    if user_df is None:
        st.error("❌ Data belum berhasil dimuat, tidak bisa dilatih. Periksa kembali file CSV Anda.")
    else:
        with st.spinner(
            "🔄 Melatih ARIMA & BiLSTM dengan data yang diupload... "
            "Mohon tunggu, jangan tutup halaman ini."
        ):
            try:
                new_results, new_forecast_df = train_models_on_upload(
                    user_df, user_page_name, fast_mode=fast_mode
                )
                st.session_state["retrained_results"] = new_results
                st.session_state["retrained_forecast_df"] = new_forecast_df
                st.session_state["retrained_page_name"] = user_page_name
                st.success(
                    "✅ Training selesai! Lihat hasilnya di tab **Perbandingan Prediksi**, "
                    "**Evaluasi Model**, dan **Rekomendasi**."
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ Training gagal: {e}")

# ── Gunakan hasil retraining (jika ada) sebagai pengganti hasil default ────
using_retrained = False
if st.session_state.get("retrained_results") is not None:
    results = st.session_state["retrained_results"]
    forecast_df = st.session_state["retrained_forecast_df"]
    if user_page_name is None:
        user_page_name = st.session_state.get("retrained_page_name")
    using_retrained = True

if results is None or forecast_df is None:
    st.warning(
        "⚠️ File hasil training (`outputs/results.json` dan `outputs/forecast_data.csv`) "
        "belum ditemukan. Jalankan notebook `Training_ARIMA_BiLSTM.ipynb` di Google Colab "
        "terlebih dahulu, lalu letakkan hasilnya di folder `outputs/`. Atau, upload CSV di "
        "sidebar dan gunakan tombol **Latih Ulang** untuk melatih model langsung di sini.\n\n"
        "Di bawah ini ditampilkan **data demo** sebagai contoh tampilan dashboard."
    )
    # ── Demo data fallback ──────────────────────────────────────────────────
    rng = pd.date_range("2023-01-01", periods=550, freq="D")
    np.random.seed(7)
    base = 9000 + np.linspace(0, 3000, len(rng))
    seasonal = 800 * np.sin(np.arange(len(rng)) * (2 * np.pi / 7))
    noise = np.random.normal(0, 400, len(rng))
    actual = np.clip(base + seasonal + noise, 0, None)

    split_idx = int(len(rng) * 0.8)
    forecast_df = pd.DataFrame({"Date": rng, "Actual": actual})
    forecast_df["ARIMA_Pred"] = np.nan
    forecast_df["BiLSTM_Pred"] = np.nan
    test_actual = actual[split_idx:]
    forecast_df.loc[split_idx:, "ARIMA_Pred"] = test_actual + np.random.normal(0, 600, len(test_actual))
    forecast_df.loc[split_idx:, "BiLSTM_Pred"] = test_actual + np.random.normal(0, 350, len(test_actual))
    forecast_df["DataSplit"] = ["Train"] * split_idx + ["Test"] * (len(rng) - split_idx)

    arima_m = compute_metrics(test_actual, forecast_df["ARIMA_Pred"].iloc[split_idx:])
    bilstm_m = compute_metrics(test_actual, forecast_df["BiLSTM_Pred"].iloc[split_idx:])
    results = {
        "page_name": "DEMO — Portal Berita Online (contoh)",
        "n_total": len(rng),
        "n_train": split_idx,
        "n_test": len(rng) - split_idx,
        "arima": {"order": [2, 1, 2], "metrics": arima_m, "ljung_box_pvalue": 0.34},
        "bilstm": {"units": 64, "lookback": 14, "dropout": 0.2, "batch_size": 16, "epochs_run": 42, "metrics": bilstm_m},
        "best_model": "ARIMA" if arima_m["RMSE"] < bilstm_m["RMSE"] else "BiLSTM",
        "adf_pvalue": 0.01,
    }
    is_demo = True
else:
    is_demo = False

if using_retrained:
    st.info(
        f"🔄 Tab di bawah menampilkan hasil **training langsung di dashboard** "
        f"untuk data: **{user_page_name}** (bukan hasil dari Colab). "
        f"Gunakan tombol **Reset** di sidebar untuk kembali ke hasil training awal."
    )


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Data Historis",
    "🔀 Perbandingan Prediksi",
    "📋 Evaluasi Model",
    "🏆 Rekomendasi",
])

forecast_df = forecast_df.sort_values("Date").reset_index(drop=True)
train_df = forecast_df[forecast_df["DataSplit"] == "Train"]
test_df = forecast_df[forecast_df["DataSplit"] == "Test"]


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA HISTORIS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    page_label = user_page_name if user_df is not None else results["page_name"]
    st.markdown(f"#### Trafik Harian — *{page_label}*")

    # Pilih sumber untuk plot: data upload jika ada, kalau tidak pakai hasil training
    plot_df = user_df if user_df is not None else forecast_df[["Date", "Actual"]].rename(columns={"Actual": "Visits"})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hari", f"{len(plot_df):,}")
    with col2:
        st.metric("Rata-rata Trafik", f"{plot_df['Visits'].mean():,.0f}")
    with col3:
        st.metric("Trafik Maksimum", f"{plot_df['Visits'].max():,.0f}")
    with col4:
        st.metric("Trafik Minimum", f"{plot_df['Visits'].min():,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["Visits"],
        mode="lines", name="Trafik Harian",
        line=dict(color="#028090", width=1.5),
        fill="tozeroy", fillcolor="rgba(2,128,144,0.08)"
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Pengunjung (Visits)",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Lihat data mentah"):
        st.dataframe(plot_df, use_container_width=True, height=300)

    if user_df is None:
        st.markdown(
            '<div class="info-box">💡 Ini menampilkan data historis dari hasil training. '
            'Upload CSV trafik baru di sidebar untuk melihat visualisasi data lain, '
            'dan opsional latih ulang model langsung di dashboard.</div>',
            unsafe_allow_html=True
        )
    elif not using_retrained:
        st.markdown(
            '<div class="info-box">💡 Ini menampilkan data historis dari file yang Anda upload. '
            'Grafik pada tab lain masih memakai hasil model dari Colab. Tekan tombol '
            '<b>🚀 Latih Ulang dengan Data Ini</b> di sidebar jika ingin melatih ARIMA & BiLSTM '
            'langsung memakai data ini.</div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERBANDINGAN PREDIKSI
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if is_demo:
        st.info("📌 Menampilkan **data demo**. Hasil sebenarnya akan muncul setelah file training diletakkan di folder `outputs/`.")

    st.markdown("#### Grafik Perbandingan: Aktual vs ARIMA vs BiLSTM")

    view_option = st.radio(
        "Tampilkan:",
        ["Seluruh Data (Train + Test)", "Hanya Data Uji (Test)"],
        horizontal=True,
    )

    if view_option == "Hanya Data Uji (Test)":
        plot_data = test_df
    else:
        plot_data = forecast_df

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=plot_data["Date"], y=plot_data["Actual"],
        mode="lines", name="Aktual",
        line=dict(color="#1E293B", width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=plot_data["Date"], y=plot_data["ARIMA_Pred"],
        mode="lines", name="Prediksi ARIMA",
        line=dict(color="#F97316", width=2, dash="dash")
    ))
    fig2.add_trace(go.Scatter(
        x=plot_data["Date"], y=plot_data["BiLSTM_Pred"],
        mode="lines", name="Prediksi BiLSTM",
        line=dict(color="#02C39A", width=2, dash="dash")
    ))

    if view_option == "Seluruh Data (Train + Test)" and len(test_df) > 0:
        fig2.add_vline(
            x=test_df["Date"].iloc[0],
            line_width=1, line_dash="dot", line_color="#94A3B8",
            annotation_text="Mulai Data Uji", annotation_position="top"
        )

    fig2.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Pengunjung (Visits)",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig2.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Grafik Residual (Selisih Aktual - Prediksi) pada Data Uji")
    res_arima = test_df["Actual"] - test_df["ARIMA_Pred"]
    res_bilstm = test_df["Actual"] - test_df["BiLSTM_Pred"]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=test_df["Date"], y=res_arima, mode="lines", name="Residual ARIMA", line=dict(color="#F97316")))
    fig3.add_trace(go.Scatter(x=test_df["Date"], y=res_bilstm, mode="lines", name="Residual BiLSTM", line=dict(color="#02C39A")))
    fig3.add_hline(y=0, line_width=1, line_color="#64748B")
    fig3.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tanggal",
        yaxis_title="Residual",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig3.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig3.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUASI MODEL
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if is_demo:
        st.info("📌 Menampilkan **data demo**. Hasil sebenarnya akan muncul setelah file training diletakkan di folder `outputs/`.")

    arima_metrics = results["arima"]["metrics"]
    bilstm_metrics = results["bilstm"]["metrics"]
    has_smape = "SMAPE" in arima_metrics and "SMAPE" in bilstm_metrics

    st.markdown("#### Tabel Perbandingan Metrik Evaluasi")

    metric_rows = ["RMSE", "MAE", "MAPE (%)"]
    arima_vals = [
        f"{arima_metrics['RMSE']:,.2f}",
        f"{arima_metrics['MAE']:,.2f}",
        f"{arima_metrics['MAPE']:,.2f}%",
    ]
    bilstm_vals = [
        f"{bilstm_metrics['RMSE']:,.2f}",
        f"{bilstm_metrics['MAE']:,.2f}",
        f"{bilstm_metrics['MAPE']:,.2f}%",
    ]
    if has_smape:
        metric_rows.append("SMAPE (%)")
        arima_vals.append(f"{arima_metrics['SMAPE']:,.2f}%")
        bilstm_vals.append(f"{bilstm_metrics['SMAPE']:,.2f}%")

    metric_table = pd.DataFrame({"Metrik": metric_rows, "ARIMA": arima_vals, "BiLSTM": bilstm_vals})
    st.dataframe(metric_table, use_container_width=True, hide_index=True)

    if max(arima_metrics.get("MAPE", 0), bilstm_metrics.get("MAPE", 0)) > 100:
        st.markdown(
            '<div class="info-box">ℹ️ Nilai <b>MAPE di atas 100%</b> pada salah satu model bukan berarti '
            'model gagal total — ini kelemahan matematis MAPE saat nilai aktual pada sebagian periode uji '
            'sangat kecil (mis. trafik anjlok tajam), sehingga error kecil sekalipun menghasilkan persentase '
            'besar. <b>SMAPE</b> lebih tahan terhadap kondisi ini karena pembaginya adalah rata-rata nilai '
            'aktual dan prediksi, bukan nilai aktual saja — gunakan SMAPE sebagai pembanding yang lebih adil '
            'di kondisi ini.</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    n_cols = 4 if has_smape else 3
    cols = st.columns(n_cols)

    def metric_delta(metric_name):
        a, b = arima_metrics[metric_name], bilstm_metrics[metric_name]
        diff = a - b
        pct = (diff / a * 100) if a != 0 else 0
        winner = "BiLSTM" if b < a else "ARIMA"
        return winner, abs(pct)

    with cols[0]:
        w, pct = metric_delta("RMSE")
        st.metric("RMSE Terbaik", w, delta=f"{pct:.1f}% lebih rendah", delta_color="normal")
    with cols[1]:
        w, pct = metric_delta("MAE")
        st.metric("MAE Terbaik", w, delta=f"{pct:.1f}% lebih rendah", delta_color="normal")
    with cols[2]:
        w, pct = metric_delta("MAPE")
        st.metric("MAPE Terbaik", w, delta=f"{pct:.1f}% lebih rendah", delta_color="normal")
    if has_smape:
        with cols[3]:
            w, pct = metric_delta("SMAPE")
            st.metric("SMAPE Terbaik", w, delta=f"{pct:.1f}% lebih rendah", delta_color="normal")

    st.write("")
    st.markdown("#### Grafik Batang Perbandingan Metrik")

    fig4 = go.Figure()
    metric_names = ["RMSE", "MAE", "MAPE", "SMAPE"] if has_smape else ["RMSE", "MAE", "MAPE"]
    fig4.add_trace(go.Bar(
        x=metric_names,
        y=[arima_metrics[m] for m in metric_names],
        name="ARIMA", marker_color="#F97316"
    ))
    fig4.add_trace(go.Bar(
        x=metric_names,
        y=[bilstm_metrics[m] for m in metric_names],
        name="BiLSTM", marker_color="#02C39A"
    ))
    fig4.update_layout(
        height=380,
        barmode="group",
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        yaxis_title="Nilai Metrik",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig4.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig4, use_container_width=True)
    if has_smape:
        st.caption(
            "💡 RMSE & MAE dalam satuan kunjungan (skala besar), sedangkan MAPE & SMAPE dalam "
            "persen (skala kecil) — grafik di atas menggabungkan keduanya untuk ringkas; untuk "
            "perbandingan skala yang presisi, lihat tabel metrik di atas."
        )

    st.write("")
    st.markdown("#### Detail Parameter Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ARIMA**")
        order = results["arima"]["order"]
        st.markdown(
            f"- Orde model: **ARIMA({order[0]}, {order[1]}, {order[2]})**\n"
            f"- p-value uji ADF (stasioneritas): **{results['adf_pvalue']:.4f}**\n"
            f"- p-value uji Ljung-Box (residual): **{results['arima']['ljung_box_pvalue']:.4f}**"
        )
    with col2:
        st.markdown("**BiLSTM**")
        bl = results["bilstm"]
        st.markdown(
            f"- Jumlah unit BiLSTM: **{bl['units']}**\n"
            f"- Lookback (window size): **{bl['lookback']} hari**\n"
            f"- Dropout rate: **{bl['dropout']}**\n"
            f"- Batch size: **{bl['batch_size']}**\n"
            f"- Epoch yang dijalankan: **{bl['epochs_run']}** (Early Stopping)"
        )

    st.write("")
    st.markdown("#### Informasi Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{results['n_total']} hari")
    with col2:
        st.metric("Data Training (80%)", f"{results['n_train']} hari")
    with col3:
        st.metric("Data Testing (20%)", f"{results['n_test']} hari")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — REKOMENDASI
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    if is_demo:
        st.info("📌 Menampilkan **data demo**. Hasil sebenarnya akan muncul setelah file training diletakkan di folder `outputs/`.")

    best = results["best_model"]
    arima_metrics = results["arima"]["metrics"]
    bilstm_metrics = results["bilstm"]["metrics"]
    best_metrics = arima_metrics if best == "ARIMA" else bilstm_metrics
    other = "BiLSTM" if best == "ARIMA" else "ARIMA"
    other_metrics = bilstm_metrics if best == "ARIMA" else arima_metrics

    improvement = (other_metrics["RMSE"] - best_metrics["RMSE"]) / other_metrics["RMSE"] * 100 if other_metrics["RMSE"] != 0 else 0
    has_smape = "SMAPE" in best_metrics and "SMAPE" in other_metrics
    mape_unreliable = best_metrics["MAPE"] > 100 or other_metrics["MAPE"] > 100

    st.markdown("#### Kesimpulan Otomatis")

    # Kategori akurasi: pakai SMAPE kalau MAPE tidak reliable (meledak akibat
    # nilai aktual yang sangat kecil di sebagian periode uji), jika tersedia.
    acc_metric_name = "SMAPE" if (mape_unreliable and has_smape) else "MAPE"
    acc_value = best_metrics[acc_metric_name]
    accuracy_label = "sangat akurat" if acc_value < 10 else "cukup akurat" if acc_value < 20 else "kurang akurat"

    caveat_html = ""
    if mape_unreliable:
        smape_line = (
            f"Sebagai pembanding yang lebih tahan terhadap kondisi ini, nilai <b>SMAPE model {best} "
            f"= {best_metrics['SMAPE']:.2f}%</b>."
            if has_smape else ""
        )
        caveat_html = f"""
        <p style="font-size:0.92rem; color:#475569;">⚠️ <b>Catatan:</b> MAPE di atas 100% pada evaluasi ini
        mengindikasikan ada periode di data uji dengan nilai trafik aktual yang sangat kecil (mis. trafik
        anjlok tajam setelah puncak), sehingga sedikit selisih prediksi menghasilkan persentase error yang
        besar secara matematis — bukan berarti model gagal total. {smape_line}</p>
        """

    st.markdown(f"""
    <div class="recommendation-box">
        <h4 style="margin-top:0; color:#028090;">🏆 Model Terbaik: {best}</h4>
        <p>Berdasarkan hasil evaluasi pada data uji, model <b>{best}</b> menghasilkan performa terbaik
        dengan nilai <b>RMSE = {best_metrics['RMSE']:,.2f}</b>, <b>MAE = {best_metrics['MAE']:,.2f}</b>,
        dan <b>MAPE = {best_metrics['MAPE']:.2f}%</b>.</p>
        <p>Nilai RMSE model {best} sekitar <b>{improvement:.1f}% lebih rendah</b> dibandingkan model {other}
        ({other_metrics['RMSE']:,.2f}), yang berarti rata-rata kesalahan prediksi model {best} lebih kecil.</p>
        <p>Dengan nilai {acc_metric_name} sebesar {acc_value:.2f}%, model ini dikategorikan
        <b>{accuracy_label}</b> {"(" + acc_metric_name + " < 10%)" if acc_value < 10 else ""}.</p>
        {caveat_html}
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Rekomendasi untuk Pengelola Website")

    if best == "ARIMA":
        st.markdown("""
- **Gunakan ARIMA sebagai model utama** untuk prediksi trafik harian, karena performanya lebih stabil pada data ini.
- ARIMA cocok jika trafik website didominasi oleh pola **musiman dan tren yang relatif linier** (misalnya pola mingguan hari kerja vs akhir pekan).
- ARIMA memiliki **keunggulan komputasi** — waktu training lebih cepat dan model lebih mudah diinterpretasikan, sehingga cocok untuk pembaruan model secara berkala (retraining harian/mingguan).
- BiLSTM tetap dapat digunakan sebagai **model pembanding** terutama saat terjadi anomali trafik (event besar, berita viral) yang sulit ditangkap pola linier.
        """)
    else:
        st.markdown("""
- **Gunakan BiLSTM sebagai model utama** untuk prediksi trafik harian, karena lebih mampu menangkap pola **non-linier dan fluktuasi tajam** pada data trafik website berita online.
- BiLSTM sangat direkomendasikan jika trafik website sering mengalami **lonjakan akibat berita viral, event nasional, atau hari libur** yang sulit dimodelkan secara linier.
- Perlu diperhatikan: BiLSTM membutuhkan **waktu training lebih lama** dan data historis yang cukup banyak agar tidak overfitting.
- ARIMA tetap berguna sebagai **baseline cepat** untuk validasi awal sebelum melakukan training ulang BiLSTM.
        """)

    st.write("")
    st.markdown("#### Implikasi untuk Manajemen Kapasitas Server")
    st.markdown(f"""
    <div class="info-box">
    Dengan prediksi trafik harian yang dihasilkan oleh model <b>{best}</b>, pengelola website dapat:
    <ul>
        <li>Menjadwalkan <b>auto-scaling server</b> sebelum lonjakan trafik terjadi (misalnya 1-2 hari menjelang hari libur).</li>
        <li>Mengurangi <b>biaya infrastruktur</b> dengan menurunkan kapasitas server saat prediksi trafik rendah.</li>
        <li>Memantau <b>akurasi model secara berkala</b> melalui dashboard ini dan melakukan retraining jika MAPE meningkat signifikan.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Ringkasan Perbandingan Akhir")
    summary_rows = ["RMSE (lebih kecil lebih baik)", "MAE (lebih kecil lebih baik)", "MAPE (lebih kecil lebih baik)", "Kategori Akurasi (MAPE)"]
    arima_summary = [
        f"{arima_metrics['RMSE']:,.2f}",
        f"{arima_metrics['MAE']:,.2f}",
        f"{arima_metrics['MAPE']:.2f}%",
        "Sangat Akurat" if arima_metrics["MAPE"] < 10 else "Cukup Akurat" if arima_metrics["MAPE"] < 20 else "Kurang Akurat",
    ]
    bilstm_summary = [
        f"{bilstm_metrics['RMSE']:,.2f}",
        f"{bilstm_metrics['MAE']:,.2f}",
        f"{bilstm_metrics['MAPE']:.2f}%",
        "Sangat Akurat" if bilstm_metrics["MAPE"] < 10 else "Cukup Akurat" if bilstm_metrics["MAPE"] < 20 else "Kurang Akurat",
    ]
    if has_smape:
        summary_rows += ["SMAPE (lebih kecil lebih baik)", "Kategori Akurasi (SMAPE)"]
        arima_summary += [
            f"{arima_metrics['SMAPE']:.2f}%",
            "Sangat Akurat" if arima_metrics["SMAPE"] < 10 else "Cukup Akurat" if arima_metrics["SMAPE"] < 20 else "Kurang Akurat",
        ]
        bilstm_summary += [
            f"{bilstm_metrics['SMAPE']:.2f}%",
            "Sangat Akurat" if bilstm_metrics["SMAPE"] < 10 else "Cukup Akurat" if bilstm_metrics["SMAPE"] < 20 else "Kurang Akurat",
        ]
    summary_df = pd.DataFrame({"Aspek": summary_rows, "ARIMA": arima_summary, "BiLSTM": bilstm_summary})
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dashboard ini dikembangkan menggunakan **Python** dan **Streamlit** sebagai bagian dari "
    "Skripsi *Analisis Perbandingan ARIMA dan BiLSTM untuk Peramalan Trafik Website Berita Online "
    "Berbasis Dashboard* — Salsabil Zahra (4123049), STMIK Al Muslim Bekasi, 2026."
)
