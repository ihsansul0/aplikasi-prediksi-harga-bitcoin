import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf

st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="₿", layout="wide")

# --- Fungsi HANYA untuk memuat model (lebih sederhana) ---
@st.cache_resource
def load_model_files():
    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    lstm_model.load_weights('models/lstm_model_weights.weights.h5')
    rf_model = joblib.load('models/random_forest_model.pkl')
    return lstm_model, rf_model

# --- Memuat data secara langsung di skrip utama ---
try:
    df = yf.download("BTC-USD", start="2020-01-01", interval="1d")
    df.dropna(inplace=True)
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])
except Exception as e:
    df = pd.DataFrame() # Jika gagal, buat DataFrame kosong

# --- Aplikasi Utama ---
st.title('📈 Aplikasi Prediksi Harga Bitcoin')
st.write("Aplikasi ini membandingkan kinerja model untuk memprediksi harga Bitcoin.")

if df.empty or len(df) < 60:
    st.error("Gagal memuat data dari yfinance. Periksa koneksi internet Anda dan refresh halaman.")
else:
    model_lstm, model_rf = load_model_files()

    st.sidebar.header('Pengaturan')
    model_selection = st.sidebar.selectbox("Pilih Model:", ("LSTM", "Random Forest"))
    
    st.header('Data Harga Bitcoin (Live)')
    st.line_chart(df['Close'])
    
    st.header(f'Hasil Prediksi Menggunakan Model {model_selection}')

    # Membuat dan melatih scaler secara live
    data_close = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_close)
    
    if st.sidebar.button('Buat Prediksi'):
        last_60_days = data_close.iloc[-60:].values
        
        if np.isnan(last_60_days).any():
            st.error("Data input mengandung nilai tidak valid (NaN).")
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