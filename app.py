# Import library utama
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

# Fungsi untuk memuat aset
@st.cache_resource
def load_keras_model():
    """Memuat model LSTM yang sudah dilatih."""
    model = load_model('models/lstm_model.keras')
    return model

@st.cache_resource
def load_rf_model():
    """Memuat model Random Forest yang sudah dilatih."""
    model = joblib.load('models/rf_model.pkl')
    return model

@st.cache_resource
def load_scaler():
    """Memuat scaler yang sudah di-fit."""
    scaler = joblib.load('models/scaler.pkl')
    return scaler

@st.cache_data(ttl="15m")
def load_data():
    """Memuat data dari CSV, membersihkannya, lalu mengambil data terbaru dari yfinance."""
    # Bagian 1: Memuat dan membersihkan data dasar dari CSV
    file_path = 'data/BTC-USD_2020-01-01_to_2025-07-01.csv'
    df_base = pd.read_csv(file_path, index_col=0)
    
    junk_rows_to_drop = ['Ticker', 'Date']
    for label in junk_rows_to_drop:
        try:
            df_base.drop(label, inplace=True)
        except KeyError:
            pass
    for col in df_base.columns:
        df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
    df_base.dropna(inplace=True)
    df_base.index = pd.to_datetime(df_base.index)
    df_base.index.name = 'Date'
    
    # Bagian 2: Mengambil data terbaru dari yfinance
    latest_data = yf.download("BTC-USD", period="10d", interval="1d")
    
    # Menyamakan format kolom dari yfinance
    if isinstance(latest_data.columns, pd.MultiIndex):
        latest_data.columns = latest_data.columns.droplevel(1)
        
    latest_data.index = pd.to_datetime(latest_data.index)
    latest_data.index.name = 'Date'
    if 'Adj Close' in latest_data.columns:
        latest_data = latest_data.drop(columns=['Adj Close'])
    
    # Bagian 3: Menggabungkan dan finalisasi
    df_combined = pd.concat([df_base, latest_data])
    df_cleaned = df_combined[~df_combined.index.duplicated(keep='last')]
    df_cleaned.sort_index(inplace=True)
    
    return df_cleaned

# Memuat semua aset yang diperlukan di awal
model_lstm = load_keras_model()
model_rf = load_rf_model()
scaler = load_scaler()
df = load_data()

# Header Aplikasi
st.title('📈 Aplikasi Prediksi Harga Bitcoin')
st.write("""
Aplikasi ini membandingkan kinerja model LSTM dan Random Forest untuk memprediksi 
harga penutupan (*Close*) Bitcoin untuk hari berikutnya. Data diperbarui secara otomatis.
""")

# Sidebar untuk Kontrol Pengguna
st.sidebar.header('Pengaturan Pengguna')

model_selection = st.sidebar.selectbox(
    "Pilih Model Prediksi:",
    ("LSTM", "Random Forest")
)

# Tampilan Body
st.header('Visualisasi Data Harga Bitcoin (Live)')
st.line_chart(df['Close'])
    
st.header(f'Hasil Prediksi Menggunakan Model {model_selection}')

if st.sidebar.button('Buat Prediksi Harga Besok'):
    # Logika Prediksi
    last_60_days = df['Close'].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    
    if model_selection == "LSTM":
        X_pred = np.array([last_60_days_scaled])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        pred_price_scaled = model_lstm.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price_scaled)[0][0]
    else: # Random Forest
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

# Expander untuk Metrik Kinerja
with st.expander("Lihat Detail Kinerja Historis Model (Evaluasi pada Data Uji)"):
    # Nilai ini diambil dari hasil pengujian di notebook EDA
    lstm_mae = 2904.64
    lstm_rmse = 3687.65
    rf_mae = 17263.62
    rf_rmse = 22732.06
    
    st.write("""
    Metrik berikut menunjukkan performa masing-masing model saat diuji dengan data historis
    yang belum pernah dilihat sebelumnya. **Semakin kecil nilainya, semakin baik.**
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
    * **MAE (Mean Absolute Error):** Ini adalah rata-rata kesalahan prediksi dalam Dolar. Jika MAE adalah 500 Dolar, artinya secara rata-rata, tebakan model meleset sebesar 500 Dolar dari harga sebenarnya.
    * **RMSE (Root Mean Squared Error):** Mirip dengan MAE, namun memberi "hukuman" yang jauh lebih besar untuk tebakan yang meleset sangat jauh.
    """)

# Footer
st.write("---")
st.write("Skripsi oleh Ihsan Sulaeman Nuraziz (21110612)")