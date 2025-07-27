# 1. Import library yang dibutuhkan
import yfinance as yf
import pandas as pd
import os

# 2. Menentukan parameter untuk data yang akan diambil
ticker = "BTC-USD"
start_date = "2020-01-01"
end_date = "2025-07-01"
nama_file_csv = f"data/{ticker}_{start_date}_to_{end_date}.csv"

# 3. Membuat folder 'data' jika belum ada secara otomatis
if not os.path.exists('data'):
    os.makedirs('data')

# 4. Mengambil data dari Yahoo Finance
print(f"Mengunduh data untuk {ticker}...")
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("Gagal mengunduh data. Tidak ada data yang diterima dari yfinance.")
    else:
        # 5. Menyimpan data ke dalam file CSV
        data.to_csv(nama_file_csv)
        print(f"Data berhasil diunduh!")
        print(f"Total {len(data)} baris data tersimpan di: {nama_file_csv}")
        print("Cuplikan data:")
        print(data.head()) # Menampilkan 5 baris pertama
        
except Exception as e:
    print(f"Terjadi error: {e}")