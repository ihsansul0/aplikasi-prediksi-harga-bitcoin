# =================================================================================
# FINAL APP.PY SCRIPT (VERSI DEBUG MENDALAM)
# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import yfinance as yf

st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="₿", layout="wide")

@st.cache_resource
def build_and_load_lstm_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.load_weights('models/lstm_model_weights.weights.h5')
    return model

@st.cache_resource
def load_rf_model():
    return joblib.load('models/random_forest_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('models/scaler.pkl')

@st.cache_data(ttl="15m")
def load_data():
    try:
        data = yf.download("BTC-USD", start="2020-01-01", interval="1d")
        if data.empty:
            raise ValueError("Data dari yfinance kosong.")
        data.dropna(inplace=True)
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])
        return data
    except Exception as e:
        st.warning(f"Gagal mengambil data live: {e}. Mencoba memuat dari file cadangan...")
        try:
            file_path = 'data/BTC-USD_2020-01-01_to_2025-07-01.csv'
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            data.dropna(inplace=True)
            return data
        except Exception:
            return pd.DataFrame()

model_lstm = build_and_load_lstm_model()
model_rf = load_rf_model()
scaler = load_scaler()
df = load_data()

st.title('📈 Aplikasi Prediksi Harga Bitcoin')
st.write("Aplikasi ini membandingkan kinerja model untuk memprediksi harga Bitcoin.")

if df.empty or len(df) < 60:
    st.error("Gagal memuat data yang cukup untuk prediksi. Periksa koneksi internet Anda dan refresh halaman.")
else:
    st.sidebar.header('Pengaturan')
    model_selection = st.sidebar.selectbox("Pilih Model:", ("LSTM", "Random Forest"))
    
    st.header('Data Harga Bitcoin (Live)')
    st.line_chart(df['Close'])
    
    st.header(f'Hasil Prediksi Menggunakan Model {model_selection}')
    
    if st.sidebar.button('Buat Prediksi'):
        st.subheader("--- PROSES DEBUG PREDIKSI ---")
        try:
            # 1. Mengambil data input
            last_60_days = df['Close'].values[-60:]
            st.write("1. Data 60 hari terakhir (last_60_days):")
            st.dataframe(pd.DataFrame(last_60_days, columns=['Close']))
            st.info(f"Apakah ada NaN di input? -> {np.isnan(last_60_days).any()}")

            # 2. Scaling data
            last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
            st.write("2. Data setelah di-scale:")
            st.dataframe(pd.DataFrame(last_60_days_scaled, columns=['Scaled Close']))

            # 3. Prediksi
            if model_selection == "LSTM":
                X_pred = np.reshape(last_60_days_scaled, (1, 60, 1))
                pred_price_scaled = model_lstm.predict(X_pred)
            else:
                X_pred = last_60_days_scaled.flatten().reshape(1, -1)
                pred_price_scaled = model_rf.predict(X_pred)
            st.write("3. Hasil prediksi (scaled):", pred_price_scaled)

            # 4. Inverse transform
            predicted_price = scaler.inverse_transform(pred_price_scaled.reshape(-1, 1))[0][0]
            st.write("4. Hasil prediksi (USD):", predicted_price)

            # 5. Mengambil harga terakhir
            last_price = df['Close'].iloc[-1]
            st.write("5. Harga terakhir (USD):", last_price)

            # 6. Menghitung perubahan
            price_change = predicted_price - last_price
            st.write("6. Perubahan harga (USD):", price_change)
            
            st.success("--- DEBUG SELESAI, MENCOBA MENAMPILKAN METRIK ---")
            st.metric(label="Prediksi Harga Besok", value=f"${predicted_price:,.2f}", delta=price_change)
            
        except Exception as e:
            st.error("Terjadi error di dalam blok prediksi:")
            st.exception(e)
    else:
        st.info(f'Tekan tombol "Buat Prediksi" untuk melihat hasilnya.')