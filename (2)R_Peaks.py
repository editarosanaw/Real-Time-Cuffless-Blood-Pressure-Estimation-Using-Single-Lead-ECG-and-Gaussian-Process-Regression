import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus

# === 1. Load dataset ===
file_path = r"C:\Users\icha kamila\Downloads\Skripsi\Percobaan_Pantompkins\PT-Data-Baru\Data\part_1.mat"
mat = loadmat(file_path)

# Mencari key utama 
keys = [k for k in mat.keys() if not k.startswith("__")]
main_key = keys[0]
signal_all = mat[main_key]

# === 2. Setup ===
fs = 125  # sampling rate
detector = Pan_Tompkins_Plus_Plus()

# Folder hasil
os.makedirs("./results", exist_ok=True)

all_results = []
summary_results = []

# === 3. Looping pendeteksian semua record ===
for i in range(signal_all.shape[1]):
    record = signal_all[0, i]  # ambil satu record (matrix 3 x N)
    if record.shape[0] < 3:
        print(f"Record {i+1} tidak lengkap (kurang channel), dilewati.")
        continue

    # Ambil ECG (baris ke-2 -> index 2 berarti channel ke-3)
    ecg_raw = record[2, :]

    # Deteksi R-peak
    r_peaks = detector.rpeak_detection(ecg_raw, fs)
    r_peaks = np.array(r_peaks).astype(int)

    refined_peaks = []
    window = int(0.05 * fs)  # 50 ms window
    for p in r_peaks:
        start = max(p - window, 0)
        end = min(p + window, len(ecg_raw))
        local_max = np.argmax(ecg_raw[start:end]) + start
        refined_peaks.append(local_max)
    r_peaks = np.array(refined_peaks)


    # Simpan detail ke list
    rpeak_time = r_peaks / fs
    for idx, t in zip(r_peaks, rpeak_time):
        all_results.append([i+1, idx, t])  # record_id, index, time(s)

    # Simpan summary (jumlah R-peak)
    summary_results.append([i+1, len(r_peaks)])

    # Simpan grafik hanya untuk 20 record pertama
    if i < 20:
        time = np.arange(len(ecg_raw)) / fs  # buat array waktu (detik)

        plt.figure(figsize=(12, 4))
        plt.plot(time, ecg_raw, label="ECG Signal")
        plt.plot(time[r_peaks], ecg_raw[r_peaks], "ro", label="R-peaks")
        plt.title(f"ECG R-peak Detection - Record {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()

        out_path = f"./results/record_{i+1}_rpeaks.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f">> Grafik record {i+1} disimpan ke {out_path}")

    print(f">> Record {i+1} selesai, {len(r_peaks)} R-peaks terdeteksi.")

# === 4. Menyimpan hasil detail ke CSV ===
df = pd.DataFrame(all_results, columns=["Record_ID", "Index", "Time (s)"])
df.to_csv("./results/Rpeak_all_records.csv", index=False)

# === 5. Simpan summary ke CSV ===
df_summary = pd.DataFrame(summary_results, columns=["Record_ID", "Total_Rpeaks"])
df_summary.to_csv("./results/Rpeak_summary.csv", index=False)