# =================================================================================
# FINAL APP.PY SCRIPT (Metode Load Weights)
# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import yfinance as yf

# --- Konfigurasi Halaman (HARUS JADI PERINTAH st PERTAMA) ---
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

# --- Fungsi untuk memuat aset ---
@st.cache_resource
def build_and_load_lstm_model():
    """Membangun arsitektur LSTM dan memuat bobot yang sudah dilatih."""
    # 1. Bangun arsitektur yang sama persis seperti di notebook
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # 2. Muat HANYA bobotnya
    model.load_weights('models/lstm_model_weights.weights.h5')
    return model

@st.cache_resource
def load_rf_model():
    """Memuat model Random Forest yang sudah dilatih."""
    model = joblib.load('models/random_forest_model.pkl')
    return model

@st.cache_resource
def load_scaler():
    """Memuat scaler yang sudah di-fit."""
    scaler = joblib.load('models/scaler.pkl')
    return scaler

@st.cache_data(ttl="15m")
def load_data():
    """Memuat dan membersihkan data historis Bitcoin."""
    # Memuat data dasar dari file CSV
    file_path = 'data/BTC-USD_2020-01-01_to_2025-07-01.csv'
    # Pastikan kolom pertama ('Date') digunakan sebagai indeks dan langsung di-parse sebagai tanggal
    df_base = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # Membersihkan data dari file CSV jika ada nilai non-numerik
    for col in df_base.columns:
        df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
    df_base.dropna(inplace=True)
    
    # Mengambil data 10 hari terakhir dari Yahoo Finance untuk memastikan data selalu up-to-date
    latest_data = yf.download("BTC-USD", period="10d", interval="1d")
    
    # Membersihkan data dari Yahoo Finance
    if not latest_data.empty:
        latest_data.index.name = 'Date'
        if 'Adj Close' in latest_data.columns:
            latest_data = latest_data.drop(columns=['Adj Close'])
    
    # Gabungkan data historis dengan data terbaru. JANGAN gunakan ignore_index=True
    # agar indeks tanggal tetap terjaga.
    df_combined = pd.concat([df_base, latest_data])
    
    # Hapus baris dengan tanggal (indeks) yang duplikat.
    # 'keep="last"' memastikan kita menggunakan data terbaru dari yfinance jika ada tumpang tindih tanggal.
    df_cleaned = df_combined[~df_combined.index.duplicated(keep='last')]
    
    # Pastikan data diurutkan berdasarkan tanggal setelah penggabungan dan pembersihan
    df_cleaned.sort_index(inplace=True)
    
    return df_cleaned

# Memuat semua aset
model_lstm = build_and_load_lstm_model()
model_rf = load_rf_model()
scaler = load_scaler()
df = load_data()

# --- Sisa kode aplikasi (tidak ada perubahan dari sini ke bawah) ---
st.title('📈 Aplikasi Prediksi Harga Bitcoin')
st.write("Aplikasi ini membandingkan kinerja model LSTM dan Random Forest...")
# (Kode UI Anda selanjutnya)
st.sidebar.header('Pengaturan Pengguna')
model_selection = st.sidebar.selectbox("Pilih Model Prediksi:",("LSTM", "Random Forest"))
st.header('Visualisasi Data Harga Bitcoin (Live)')
st.line_chart(df['Close'])
st.header(f'Hasil Prediksi Menggunakan Model {model_selection}')
if st.sidebar.button('Buat Prediksi Harga Besok'):
    last_60_days = df['Close'].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    if model_selection == "LSTM":
        X_pred = np.array([last_60_days_scaled])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        pred_price_scaled = model_lstm.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price_scaled)[0][0]
    else:
        X_pred = np.array([last_60_days_scaled.flatten()])
        pred_price_scaled = model_rf.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price_scaled.reshape(-1, 1))[0][0]
    last_price = df['Close'].values[-1]
    price_change = predicted_price - last_price
    st.write(f"Harga Penutupan Terakhir (Live): **${last_price:,.2f}**")
    st.metric(label="Prediksi Harga untuk Besok", value=f"${predicted_price:,.2f}", delta=price_change)
    st.subheader(f"Grafik Detail Prediksi ({model_selection}) vs. Data Aktual")
    last_60_days_df = df['Close'][-60:].reset_index()
    next_day_date = last_60_days_df['Date'].iloc[-1] + pd.Timedelta(days=1)
    prediction_df = pd.DataFrame({'Date': [next_day_date], 'Close': [predicted_price]})
    plot_df = pd.concat([last_60_days_df, prediction_df]).set_index('Date')
    st.line_chart(plot_df['Close'])
else:
    st.info(f'Tekan tombol "Buat Prediksi Harga Besok" di sidebar untuk melihat hasil prediksi dari model {model_selection}.')
with st.expander("Lihat Detail Kinerja Historis Model (Evaluasi pada Data Uji)"):
    lstm_mae = 2904.64
    lstm_rmse = 3687.65
    rf_mae = 17263.62
    rf_rmse = 22732.06
    st.markdown("""---
**Penjelasan Metrik:**
* **MAE (Mean Absolute Error):** Ini adalah **rata-rata kesalahan prediksi** dalam Dolar. Jika MAE adalah $500, artinya secara rata-rata, tebakan model meleset sebesar $500 dari harga sebenarnya.
* **RMSE (Root Mean Squared Error):** Mirip dengan MAE, namun **memberi "hukuman" yang jauh lebih besar untuk tebakan yang meleset sangat jauh**.
""")
st.write("---")
st.write("Skripsi oleh Nama Anda (NIM Anda)")