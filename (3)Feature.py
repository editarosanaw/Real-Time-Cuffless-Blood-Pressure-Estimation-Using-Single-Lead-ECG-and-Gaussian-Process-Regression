import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ============================================================
# KONFIGURASI DASAR
# ============================================================
fs = 125  # Hz
file_path = r"C:\Users\icha kamila\Downloads\Skripsi\Percobaan_Pantompkins\PT-Data-Baru\Data\part_1.mat"
rpeak_csv = "./results/Rpeak_all_records.csv"
out_csv = "./results/ECG_features_summary.csv"

# ============================================================
# 1. LOAD DATASET & R-PEAK
# ============================================================
df_rpeak = pd.read_csv(rpeak_csv)
mat = loadmat(file_path)
main_key = [k for k in mat.keys() if not k.startswith("__")][0]
signal_all = mat[main_key]

features_results = []

# ============================================================
# 2. LOOP PER RECORD
# ============================================================
for rec_id, group in df_rpeak.groupby("Record_ID"):
    record = signal_all[0, rec_id - 1]
    ecg_raw = record[2, :].astype(float)

    r_peaks = group["Index"].values.astype(int)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < len(ecg_raw))]
    if len(r_peaks) < 2:
        continue

    # ---------- RR interval ----------
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 2 else np.nan
    pnn50 = np.mean(np.abs(np.diff(rr_intervals)) > 0.05) if len(rr_intervals) > 2 else np.nan
    cvrr = sdnn / mean_rr if mean_rr > 0 else np.nan

    # ---------- Mean HR ----------
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else np.nan

    # ---------- QRS duration ----------
    qrs_durations = [(min(len(ecg_raw), rp + int(0.05 * fs)) - max(0, rp - int(0.05 * fs))) / fs for rp in r_peaks]
    mean_qrs = np.nanmean(qrs_durations)

    # ---------- QT interval, T amplitude ----------
    qt_intervals, t_amps = [], []
    for rp in r_peaks:
        start = max(0, rp - int(0.05 * fs))
        end = min(len(ecg_raw), rp + int(0.6 * fs))
        seg = ecg_raw[start:end]
        if len(seg) < 5:
            continue

        # cari puncak T (setelah R)
        t_max = np.argmax(seg)
        deriv = np.gradient(seg)
        t_offset = None
        for k in range(t_max, len(deriv)):
            if abs(deriv[k]) < 0.002:  # slope datar => akhir T
                t_offset = start + k
                break

        if t_offset is not None:
            qt_intervals.append((t_offset - start) / fs)
            # amplitude T relatif terhadap baseline sekitar R
            t_start = rp + int(0.15 * fs)
            t_end = min(len(ecg_raw), rp + int(0.4 * fs))
            if t_end > t_start:
                seg_t = ecg_raw[t_start:t_end]
                baseline = np.median(ecg_raw[max(0, rp - int(0.1 * fs)):rp])
                t_amps.append(np.max(seg_t) - baseline)

    mean_qt = np.nanmean(qt_intervals)
    qtc = mean_qt / np.sqrt(mean_rr) if not np.isnan(mean_qt) and mean_rr > 0 else np.nan
    tq = mean_rr - mean_qt if mean_rr > 0 and not np.isnan(mean_qt) else np.nan
    sdi = mean_qt / tq if (tq is not None and tq > 0) else np.nan
    sdin = mean_qt / mean_rr if mean_rr > 0 else np.nan

    # ---------- Amplitudo R ----------
    r_amplitudes = ecg_raw[r_peaks]
    mean_r_amp = np.nanmean(r_amplitudes)

    # ---------- R slope up max ----------
    slopes = []
    for rp in r_peaks:
        start = max(0, rp - int(0.04 * fs))
        end = rp
        seg = ecg_raw[start:end]
        if len(seg) > 2:
            slopes.append(np.max(np.gradient(seg)))
    r_slope_up_max = np.nanmean(slopes)

    # ---------- QRS area ----------
    areas = []
    for rp in r_peaks:
        start = max(0, rp - int(0.05 * fs))
        end = min(len(ecg_raw), rp + int(0.05 * fs))
        seg = ecg_raw[start:end]
        if len(seg) > 0:
            baseline = np.median(seg)
            # integrasi numerik dari deviasi terhadap baseline
            areas.append(np.trapezoid(np.abs(seg - baseline), dx=1/fs))
    qrs_area = np.nanmean(areas)

    # ---------- T amplitude ----------
    mean_t_amp = np.nanmean(t_amps)

    # ---------- Hjorth Parameters ----------
    activity = np.var(ecg_raw)
    mobility = np.std(np.diff(ecg_raw)) / (np.std(ecg_raw) + 1e-8)
    complexity = (np.std(np.diff(np.diff(ecg_raw))) / (np.std(np.diff(ecg_raw)) + 1e-8)) / (mobility + 1e-8)

    # ---------- Statistik tambahan ----------
    rms = np.sqrt(np.mean(ecg_raw ** 2))
    cv = np.std(ecg_raw) / (np.mean(np.abs(ecg_raw)) + 1e-8)

    # ---------- Fitur kontraktilitas tambahan ----------
    contractility_index = (qrs_area * mean_r_amp) / (mean_rr + 1e-6)
    r_amp_per_rr = mean_r_amp / (mean_rr + 1e-6)
    qt_tq_ratio = qtc / (tq + 1e-6) if tq and tq > 0 else np.nan

    # ---------- Simpan semua fitur ----------
    features_results.append([
        rec_id, len(r_peaks),
        mean_rr, mean_hr, mean_qrs, mean_qt, qtc, tq, sdi, sdin,
        mean_r_amp, mean_t_amp,
        r_slope_up_max, qrs_area,
        activity, mobility, complexity,
        rms, cv,
        contractility_index, r_amp_per_rr, qt_tq_ratio
    ])

    # ---------- Visualisasi contoh ----------
    if rec_id == 8:
        time = np.arange(len(ecg_raw)) / fs
        plt.figure(figsize=(15, 5))
        plt.plot(time, ecg_raw, label="ECG", color="black")
        plt.plot(time[r_peaks], ecg_raw[r_peaks], "ro", label="R-peaks")
        for rp in r_peaks:
            q_start = max(0, rp - int(0.05 * fs))
            q_end = min(len(ecg_raw), rp + int(0.05 * fs))
            plt.axvspan(time[q_start], time[q_end], color="blue", alpha=0.2,
                        label="QRS window" if rp == r_peaks[0] else "")
        plt.title("Record 8 - Morphological + Hjorth + Contractility Features")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("./results/record8_features_plot.png", dpi=300)
        plt.show()

# ============================================================
# 3. BENTUK DATAFRAME FITUR
# ============================================================
df_features = pd.DataFrame(features_results, columns=[
    "Record_ID", "Total_Rpeaks",
    "Mean_RR(s)", "Mean_HR(bpm)", "Mean_QRS(s)", "Mean_QT(s)", "QTc_Bazett(s)",
    "TQ(s)", "SDI(QT/TQ)", "SDIn(QT/RR)",
    "R_amp", "T_amp",
    "R_slope_up_max", "QRS_area",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "RMS", "CV",
    "Contractility_Index", "R_amp_per_RR", "QT_TQ_ratio"
])

# normalisasi z-score per-record (biar siap buat model SBP)
for col in ["Contractility_Index", "R_amp_per_RR", "QRS_area"]:
    df_features[col + "_z_by_rec"] = df_features.groupby("Record_ID")[col].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6)
    )

# ============================================================
# 4. SIMPAN KE CSV
# ============================================================
df_features.to_csv(out_csv, index=False)
print(f">> Fitur ECG tersimpan ke '{out_csv}'")
print(f"Total data fitur: {len(df_features)} sampel dari {df_features['Record_ID'].nunique()} record")
