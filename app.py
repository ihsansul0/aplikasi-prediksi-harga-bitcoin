# =================================================================================
# FINAL APP.PY SCRIPT (STABLE VERSION)
# =================================================================================

# Import library utama
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
    """Memuat model Random Forest yang sudah dilatih."""
    return joblib.load('models/random_forest_model.pkl')

@st.cache_resource
def load_scaler():
    """Memuat scaler yang sudah di-fit."""
    return joblib.load('models/scaler.pkl')

@st.cache_data(ttl="15m")
def load_data():
    """
    Mencoba memuat data live. Jika gagal, memuat dari file CSV sebagai cadangan.
    """
    try:
        # Coba muat data terbaru langsung dari yfinance
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
            # Jika gagal, muat dari CSV sebagai cadangan
            file_path = 'data/BTC-USD_2020-01-01_to_2025-07-01.csv'
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            data.dropna(inplace=True)
            return data
        except Exception as e2:
            st.error(f"Gagal memuat data dari file cadangan: {e2}")
            return pd.DataFrame()

# Memuat semua aset
model_lstm = build_and_load_lstm_model()
model_rf = load_rf_model()
scaler = load_scaler()
df = load_data()

# --- Header Aplikasi ---
st.title('📈 Aplikasi Prediksi Harga Bitcoin')
st.write("Aplikasi ini membandingkan kinerja model LSTM dan Random Forest untuk memprediksi harga Bitcoin.")

# --- Aplikasi Utama ---
if df.empty:
    st.error("Gagal memuat data. Periksa koneksi internet Anda dan refresh halaman.")
else:
    # --- Sidebar ---
    st.sidebar.header('Pengaturan')
    model_selection = st.sidebar.selectbox("Pilih Model:", ("LSTM", "Random Forest"))
    
    # --- Tampilan Body ---
    st.header('Data Harga Bitcoin (Live)')
    st.line_chart(df['Close'])
    
    st.header(f'Hasil Prediksi Menggunakan Model {model_selection}')
    
    if st.sidebar.button('Buat Prediksi'):
        last_60_days = df['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        
        if model_selection == "LSTM":
            X_pred = np.reshape(last_60_days_scaled, (1, 60, 1))
            pred_price_scaled = model_lstm.predict(X_pred)
        else: # Random Forest
            X_pred = last_60_days_scaled.flatten().reshape(1, -1)
            pred_price_scaled = model_rf.predict(X_pred)

        predicted_price = scaler.inverse_transform(pred_price_scaled.reshape(-1, 1))[0][0]
        last_price = df['Close'].iloc[-1]
        price_change = predicted_price - last_price
        
        st.metric(label="Prediksi Harga Besok", value=f"${predicted_price:,.2f}", delta=f"{price_change:,.2f}")
        
        st.subheader(f"Grafik Detail Prediksi ({model_selection})")
        last_60_days_df = df['Close'][-60:].reset_index()
        next_day_date = last_60_days_df['Date'].iloc[-1] + pd.Timedelta(days=1)
        prediction_df = pd.DataFrame({'Date': [next_day_date], 'Close': [predicted_price]})
        plot_df = pd.concat([last_60_days_df, prediction_df]).set_index('Date')
        st.line_chart(plot_df['Close'])
    else:
        st.info(f'Tekan tombol "Buat Prediksi" di sidebar untuk melihat hasil dari model {model_selection}.')

    with st.expander("Lihat Detail Kinerja Historis Model"):
        lstm_mae = 2904.64
        lstm_rmse = 3687.65
        rf_mae = 17263.62
        rf_rmse = 22732.06
        
        st.markdown("""
        Metrik berikut dihitung dari performa model saat diuji menggunakan data historis yang belum pernah dilihat sebelumnya.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model LSTM")
            st.metric(label="MAE", value=f"${lstm_mae:,.2f}")
            st.metric(label="RMSE", value=f"${lstm_rmse:,.2f}")
        with col2:
            st.subheader("Model Random Forest")
            st.metric(label="MAE", value=f"${rf_mae:,.2f}")
            st.metric(label="RMSE", value=f"${rf_rmse:,.2f}")
            
        st.markdown("""
        ---
        **Penjelasan Metrik:**
        * **MAE (Mean Absolute Error):** Rata-rata kesalahan prediksi dalam Dolar.
        * **RMSE (Root Mean Squared Error):** Mirip MAE, namun memberi "hukuman" lebih besar untuk tebakan yang meleset sangat jauh.
        """)

    st.write("---")
    st.write("Skripsi oleh Nama Anda (NIM Anda)")