import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf

st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="₿", layout="wide")

# HANYA MEMUAT MODEL, KARENA SCALER DAN DATA AKAN DIBUAT SECARA LIVE
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

# FUNGSI LOAD DATA YANG PALING STABIL
@st.cache_data(ttl="15m")
def load_data():
    try:
        df = yf.download("BTC-USD", start="2020-01-01", interval="1d")
        if df.empty:
            raise ValueError("Data dari yfinance kosong.")

        # === BAGIAN PEMBERSIHAN TOTAL ===
        df.reset_index(inplace=True)
        df.drop_duplicates(subset='Date', keep='last', inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        if 'Adj Close' in df.columns:
            df = df.drop(columns=['Adj Close'])
            
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan fatal saat memuat data: {e}")
        return pd.DataFrame()

# Memuat model
model_lstm = build_and_load_lstm_model()
model_rf = load_rf_model()
# Mengambil data live
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

    # MEMBUAT SCALER BARU SECARA LIVE
    data_close = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_close) # Latih scaler pada keseluruhan data yang ada
    
    if st.sidebar.button('Buat Prediksi'):
        last_60_days = data_close.values[-60:]
        
        if np.isnan(last_60_days).any():
            st.error("Gagal membuat prediksi: Data input mengandung nilai tidak valid (NaN). Coba refresh halaman.")
        else:
            last_60_days_scaled = scaler.transform(last_60_days)
            
            if model_selection == "LSTM":
                X_pred = np.reshape(last_60_days_scaled, (1, 60, 1))
                pred_price_scaled = model_lstm.predict(X_pred)
            else:
                X_pred = last_60_days_scaled.flatten().reshape(1, -1)
                pred_price_scaled = model_rf.predict(X_pred)

            predicted_price = scaler.inverse_transform(pred_price_scaled)[0][0]
            last_price = df['Close'].iloc[-1]
            price_change = predicted_price - last_price
            
            st.metric(label="Prediksi Harga Besok", value=f"${predicted_price:,.2f}", delta=price_change)
    else:
        st.info(f'Tekan tombol "Buat Prediksi" untuk melihat hasilnya.')

    st.write("---")
    st.write("Skripsi oleh Nama Anda (NIM Anda)")