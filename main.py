import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import os
from datetime import datetime

# Fungsi untuk membuat data dummy awal untuk pelatihan model
def create_dummy_data(filename="data_dummy_kebakaran.csv"):
    if not os.path.exists(filename):
        data = {
            "CO2": [400, 1000, 450, 1200, 380, 1100, 420, 1300, 500, 900] + \
                   list(np.random.randint(350, 1400, 90)),
            "Suhu": [25, 40, 28, 45, 26, 42, 27, 48, 30, 38] + \
                    list(np.random.uniform(20, 50, 90)),
            "Kelembaban": [60, 30, 55, 25, 65, 28, 50, 20, 58, 35] + \
                         list(np.random.uniform(15, 70, 90)),
            "Label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] + \
                     [1 if x > 800 or y > 35 else 0 for x, y in zip(
                         np.random.randint(350, 1400, 90), np.random.uniform(20, 50, 90))]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        st.write(f"File {filename} berhasil dibuat.")
    else:
        st.write(f"File {filename} sudah ada.")

# Fungsi untuk melatih model
def train_model(filename="data_dummy_kebakaran.csv"):
    data = pd.read_csv(filename)
    X = data[['CO2', 'Suhu', 'Kelembaban']]
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Fungsi untuk prediksi kebakaran (diperbaiki untuk menangani feature names)
def deteksi_kebakaran(model, co2, suhu, kelembaban):
    try:
        # Membuat DataFrame dengan nama kolom yang sama seperti saat pelatihan
        input_data = pd.DataFrame([[float(co2), float(suhu), float(kelembaban)]], 
                                 columns=['CO2', 'Suhu', 'Kelembaban'])
        prediksi = model.predict(input_data)[0]
        return "Kebakaran Terdeteksi!" if prediksi == 1 else "Kondisi Normal"
    except ValueError:
        return "Error: Masukkan nilai numerik!"

# Fungsi untuk menyimpan hasil
def save_result(co2, suhu, kelembaban, result, filename="hasil_deteksi_realtime.txt"):
    with open(filename, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"CO2: {co2} ppm\n")
        f.write(f"Suhu: {suhu} °C\n")
        f.write(f"Kelembaban: {kelembaban} %\n")
        f.write(f"Hasil: {result}\n\n")
    return filename

# Fungsi untuk menghasilkan data dummy secara real-time
def generate_realtime_dummy_data():
    co2 = np.random.randint(350, 1500)  # CO2 antara 350-1500 ppm
    suhu = np.random.uniform(20, 50)    # Suhu antara 20-50°C
    kelembaban = np.random.uniform(10, 90)  # Kelembaban antara 10-90%
    return co2, suhu, kelembaban

# Streamlit app untuk pembacaan real-time
def main():
    st.title("Sistem Deteksi Kebakaran Hutan Berbasis AI (Real-Time)")
    
    # Membuat dan melatih model
    create_dummy_data()
    model, accuracy = train_model()
    st.write(f"Akurasi Model: {accuracy*100:.2f}%")
    
    st.header("Pembacaan Sensor Real-Time")
    
    # Placeholder untuk menampilkan data
    co2_placeholder = st.empty()
    suhu_placeholder = st.empty()
    kelembaban_placeholder = st.empty()
    result_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Tombol untuk memulai dan menghentikan simulasi
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.button("Mulai Simulasi"):
        st.session_state.running = True
    
    if st.button("Hentikan Simulasi"):
        st.session_state.running = False
    
    # Simulasi real-time
    while st.session_state.running:
        # Generate data dummy
        co2, suhu, kelembaban = generate_realtime_dummy_data()
        
        # Tampilkan data
        co2_placeholder.write(f"Kadar CO2 (ppm): {co2:.2f}")
        suhu_placeholder.write(f"Suhu (°C): {suhu:.2f}")
        kelembaban_placeholder.write(f"Kelembaban (%): {kelembaban:.2f}")
        
        # Prediksi
        result = deteksi_kebakaran(model, co2, suhu, kelembaban)
        result_placeholder.write(f"**Hasil**: {result}")
        
        # Simpan hasil
        saved_file = save_result(co2, suhu, kelembaban, result)
        status_placeholder.write(f"Hasil disimpan ke {saved_file}")
        
        # Tunggu 2 detik sebelum pembacaan berikutnya
        time.sleep(2)
        
        # Refresh halaman untuk simulasi real-time
        st.rerun()

if __name__ == "__main__":
    main()