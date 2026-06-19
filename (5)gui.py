import os
import time
import json
import math
import threading
from collections import deque
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
import pyqtgraph as pg
from joblib import load as joblib_load
from serial import Serial
from scipy.signal import butter, filtfilt, iirnotch

from pyshimmer import ShimmerBluetooth, DataPacket, EChannelType, DEFAULT_BAUDRATE
from algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus

# ------------------------------------------------------------
# KONFIG DASAR
# ------------------------------------------------------------
FS = 125
STREAM_DT = 0.1                   # update GUI tiap 100 ms
BUFFER_SECONDS = 30               # buffer 30 detik
FEATURE_WINDOW_SEC = 20           # window fitur 20 detik
PREDICT_INTERVAL = 10.0           # prediksi tiap 10 detik

RESULTS_DIR = "results"
MODELS_DIR = "models"
SBP_MODEL_PKL = os.path.join(MODELS_DIR, "gpr_sbp_revisi.pkl")
DBP_MODEL_PKL = os.path.join(MODELS_DIR, "gpr_dbp_revisi.pkl")
FEATURE_ORDER_JSON = os.path.join(MODELS_DIR, "features_order.json")

PORT = "COM11"                    

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# STATISTIK FITUR TRAINING 
# ------------------------------------------------------------
TRAINING_STATS = {
    "Mean_QT(s)":      {"mean": 0.16677517628000713, "std": 0.06299517293060142},
    "TQ(s)":           {"mean": 0.5694139594114122,  "std": 0.15839566289940465},
    "QTc_Bazett(s)":   {"mean": 0.19947462373196448, "std": 0.08778862864971136},
    "Hjorth_Mobility": {"mean": 0.5540765718546059,  "std": 0.1957138682586237},
    "Hjorth_Complexity": {"mean": 1.894426795663087, "std": 0.43456218794481016},
    "RMS":             {"mean": 0.557967122217131,   "std": 0.2654334826537493},
    "SDI(QT/TQ)":      {"mean": 0.3814838919454695,  "std": 0.5107865234766669},
    "SDIn(QT/RR)":     {"mean": 0.24078524359236897, "std": 0.126874361432792},
    "HR(bpm)":         {"mean": 84.00825367532502,   "std": 15.599935239848898},
    # HRV(ms): aproksimasi fisiologis
    "HRV(ms)":         {"mean": 60.0,                "std": 25.0},
}

# jumlah minimal window prediksi untuk kalibrasi domain Shimmer
DA_MIN_SAMPLES = 15  # kira-kira 15*10s = 150 detik


# ------------------------------------------------------------
# KATEGORI TEKANAN DARAH
# ------------------------------------------------------------
class Category:
    @staticmethod
    def label_and_color(sbp, dbp):
        if sbp is None or dbp is None:
            return ("-", (200, 200, 200))

        if sbp < 90 or dbp < 60:
            return ("Low", (255, 152, 0))
        if sbp < 120 and dbp < 80:
            return ("Normal", (39, 174, 96))
        if 120 <= sbp < 130 and dbp < 80:
            return ("Elevated", (52, 152, 219))
        if (130 <= sbp < 140) or (80 <= dbp < 90):
            return ("High-1", (231, 126, 35))
        if sbp >= 140 or dbp >= 90:
            return ("High-2", (231, 76, 60))

        return ("Normal", (39, 174, 96))


# ------------------------------------------------------------
# R-PEAK REFINEMENT
# ------------------------------------------------------------
def refine_rpeaks(sig_norm, r_peaks, fs, min_rr_ms=280, amp_thr_rel=0.30):
    r_peaks = np.asarray(r_peaks, dtype=int)
    if r_peaks.size == 0:
        return r_peaks

    amps = np.abs(sig_norm[r_peaks])
    max_amp = np.max(amps)
    if max_amp <= 0:
        return np.array([], dtype=int)

    amp_thr = amp_thr_rel * max_amp
    cand = r_peaks[amps >= amp_thr]
    if cand.size == 0:
        return np.array([r_peaks[np.argmax(amps)]], dtype=int)

    cand = np.sort(cand)
    min_dist = int((min_rr_ms / 1000.0) * fs)

    refined = [cand[0]]
    for p in cand[1:]:
        if p - refined[-1] < min_dist:
            if np.abs(sig_norm[p]) > np.abs(sig_norm[refined[-1]]):
                refined[-1] = p
        else:
            refined.append(p)

    return np.asarray(refined, dtype=int)


# ------------------------------------------------------------
# EKSTRAKTOR FITUR 
# ------------------------------------------------------------
class FeatureExtractorRevisi:
    def __init__(self, fs):
        self.fs = fs

    def compute(self, ecg, r_peaks):
        fs = self.fs
        ecg = np.asarray(ecg, float)
        N = len(ecg)

        r_peaks = np.asarray([p for p in r_peaks if 0 < p < N - 1], int)
        if N < fs * 3 or len(r_peaks) < 3:
            return None

        # RMS
        rms = float(np.sqrt(np.mean(ecg ** 2)))

        # RR interval
        rr = np.diff(r_peaks) / fs
        if len(rr) == 0:
            return None
        mean_rr = float(np.nanmean(rr))
        hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
        hrv = float(np.std(rr) * 1000.0) if len(rr) > 1 else np.nan

        # QT & TQ (deteksi T sederhana)
        qt_list, tq_list = [], []
        for i in range(len(r_peaks) - 1):
            r1, r2 = r_peaks[i], r_peaks[i + 1]

            search_start = r1 + int(0.15 * fs)
            search_stop = r1 + int(0.4 * fs)
            search_start = max(search_start, 0)
            search_stop = min(search_stop, N)
            if search_stop <= search_start:
                continue

            seg = ecg[search_start:search_stop]
            t_rel = int(np.argmax(seg))
            t_idx = search_start + t_rel

            qt = (t_idx - r1) / fs
            tq = (r2 - t_idx) / fs

            if qt > 0 and tq > 0:
                qt_list.append(qt)
                tq_list.append(tq)

        qt_arr = np.array(qt_list)
        tq_arr = np.array(tq_list)

        mean_qt = float(np.nanmean(qt_arr)) if qt_arr.size else np.nan
        mean_tq = float(np.nanmean(tq_arr)) if tq_arr.size else np.nan

        # QTc Bazett
        if mean_qt > 0 and mean_rr > 0:
            qtc_bazett = float(mean_qt / np.sqrt(mean_rr))
        else:
            qtc_bazett = np.nan

        # Hjorth parameters
        diff1 = np.diff(ecg)
        diff2 = np.diff(diff1)
        var0 = float(np.var(ecg))
        var1 = float(np.var(diff1))
        var2 = float(np.var(diff2))

        if var0 > 0:
            hj_mob = float(np.sqrt(var1 / var0))
        else:
            hj_mob = np.nan

        if var1 > 0:
            hj_comp = float(np.sqrt(var2 / var1))
        else:
            hj_comp = np.nan

        # SDI rasio QT/TQ, QT/RR
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_qt_tq = qt_arr / tq_arr if (qt_arr.size and tq_arr.size) else np.array([])
            ratio_qt_rr = qt_arr / rr[:len(qt_arr)] if rr.size else np.array([])

        sdi_qt_tq = float(np.nanstd(ratio_qt_tq)) if ratio_qt_tq.size else np.nan
        sdin_qt_rr = float(np.nanstd(ratio_qt_rr)) if ratio_qt_rr.size else np.nan

        feat = {
            "Mean_QT(s)": mean_qt,
            "TQ(s)": mean_tq,
            "QTc_Bazett(s)": qtc_bazett,
            "Hjorth_Mobility": hj_mob,
            "Hjorth_Complexity": hj_comp,
            "RMS": rms,
            "SDI(QT/TQ)": sdi_qt_tq,
            "SDIn(QT/RR)": sdin_qt_rr,
            "HR(bpm)": float(hr),
            "HRV(ms)": hrv,
        }
        return feat


# ------------------------------------------------------------
# PREDICTOR GPR (mean ± std) 
# ------------------------------------------------------------
class PredictorRevisi:
    def __init__(self):
        self.model_sbp = joblib_load(SBP_MODEL_PKL)
        self.model_dbp = joblib_load(DBP_MODEL_PKL)

        with open(FEATURE_ORDER_JSON, "r") as f:
            self.feature_order = json.load(f)

        print("[INFO] Fitur yang dipakai model:", self.feature_order)
        

    def predict(self, feat_dict):
        try:
            X = np.array([[feat_dict[k] for k in self.feature_order]], dtype=float)
        except KeyError as e:
            return None, None, None, None, f"Fitur hilang: {e}"

        if np.any(~np.isfinite(X)):
            return None, None, None, None, "Ada fitur NaN/inf"

        try:
            sbp_mean, sbp_std = self.model_sbp.predict(X, return_std=True)
            dbp_mean, dbp_std = self.model_dbp.predict(X, return_std=True)

            sbp_mean = float(sbp_mean[0])
            sbp_std = float(sbp_std[0])
            dbp_mean = float(dbp_mean[0])
            dbp_std = float(dbp_std[0])


            return sbp_mean, sbp_std, dbp_mean, dbp_std, None

        except Exception as e:
            return None, None, None, None, str(e)


# ------------------------------------------------------------
# THREAD SHIMMER
# ------------------------------------------------------------
class ShimmerThread(threading.Thread):
    def __init__(self, port, gui_ref):
        super().__init__(daemon=True)
        self.port = port
        self.gui = gui_ref
        self.running = False

    @staticmethod
    def adc_to_mV(raw_value, vref=4.096, gain=6):
        return (raw_value * vref / ((2 ** 23 - 1) * gain)) * 1000.0

    def stream_cb(self, pkt: DataPacket):
        try:
            if EChannelType.EXG_ADS1292R_2_CH1_24BIT in pkt._values:
                raw = pkt[EChannelType.EXG_ADS1292R_2_CH1_24BIT]
                mv = self.adc_to_mV(raw)
                self.gui.buffer.append(mv)
        except Exception as e:
            print(f"[Callback Error] {e}")

    def run(self):
        print(f"[INFO] Menghubungkan ke Shimmer di {self.port}...")
        ser = Serial(self.port, DEFAULT_BAUDRATE, timeout=None)
        shimmer = ShimmerBluetooth(ser)
        shimmer.initialize()
        shimmer.add_stream_callback(self.stream_cb)
        shimmer.start_streaming()
        self.running = True
        print("[INFO] Streaming dimulai.")

        while self.running:
            time.sleep(0.05)

        shimmer.stop_streaming()
        shimmer.shutdown()
        print("[INFO] Streaming dihentikan.")


# ------------------------------------------------------------
# GUI + DOMAIN ADAPTATION
# ------------------------------------------------------------
class ECGGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG → BP Prediction (GPR Revisi V2 + DA)")
        self.resize(1300, 780)
        self.setStyleSheet("background-color:#F0ECFF;")

        self.detector = Pan_Tompkins_Plus_Plus()
        self.extractor = FeatureExtractorRevisi(FS)
        self.predictor = PredictorRevisi()

        self.buffer = deque(maxlen=int(BUFFER_SECONDS * FS))
        self.streaming = False
        self.last_predict_ts = 0.0

        # smoothing history
        self.sbp_hist = deque(maxlen=3)
        self.dbp_hist = deque(maxlen=3)
        self.hr_hist = deque(maxlen=3)
        self.hrv_hist = deque(maxlen=3)

        # ---------- DOMAIN ADAPTATION STATE ----------
        self.da_stats = {
            name: {"n": 0, "mean": 0.0, "M2": 0.0}
            for name in TRAINING_STATS.keys()
        }
        self.da_ready = False
        self.da_total_windows = 0

        self._build_ui()
        self._init_timer()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel(
            "<span style='font-size:24px;font-weight:700;'>"
            "❤ Real-Time ECG + BP (GPR Revisi, mean ± std, DA)"
            "</span>"
        )
        root.addWidget(title)

        grid = QGridLayout()
        root.addLayout(grid)

        # Plot ECG
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("left", "Amplitude (mV)")
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve = self.plot.plot(pen=pg.mkPen("#2C7BE5", width=2))
        self.rpeak_scatter = pg.ScatterPlotItem(
            brush=pg.mkBrush(255, 99, 71), size=8
        )
        self.plot.addItem(self.rpeak_scatter)

        grid.addWidget(self.plot, 0, 0, 1, 2)

        # Info panel
        info = QVBoxLayout()
        self.value_bp = QLabel(
            "<span style='font-size:44px;font-weight:800;'>--/--</span> mmHg"
        )
        self.value_bp_std = QLabel(
            "<span style='font-size:18px;'>± -- / -- mmHg</span>"
        )
        self.value_cat = QLabel("<span style='font-size:20px;'>Kategori: -</span>")
        self.value_hr = QLabel("<span style='font-size:20px;'>HR: - bpm</span>")
        self.value_hrv = QLabel("<span style='font-size:20px;'>HRV: - ms</span>")
        self.value_rpeaks = QLabel("<span style='font-size:18px;'>R-peaks: 0</span>")
        self.value_da = QLabel(
            "<span style='font-size:16px;color:#666;'>Domain Adaptation: warming up...</span>"
        )

        for w in (self.value_bp, self.value_bp_std, self.value_cat,
                  self.value_hr, self.value_hrv, self.value_rpeaks, self.value_da):
            info.addWidget(w)

        grid.addLayout(info, 0, 2, 1, 1)

        # Buttons
        controls = QHBoxLayout()
        self.btn_start = QPushButton("Start Stream")
        self.btn_stop = QPushButton("Stop")
        for b in (self.btn_start, self.btn_stop):
            b.setFixedHeight(40)
            b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            controls.addWidget(b)
        grid.addLayout(controls, 1, 0, 1, 3)

        # Log table
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Waktu", "SBP", "Std SBP", "DBP", "Std DBP",
            "Kategori", "HR", "HRV"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.table)

        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

    def _init_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(int(STREAM_DT * 1000))

    # ---------- Streaming ----------
    def start_stream(self):
        if self.streaming:
            return
        self.buffer.clear()
        self.sbp_hist.clear()
        self.dbp_hist.clear()
        self.hr_hist.clear()
        self.hrv_hist.clear()

        # reset domain adaptation state
        self.da_stats = {
            name: {"n": 0, "mean": 0.0, "M2": 0.0}
            for name in TRAINING_STATS.keys()
        }
        self.da_ready = False
        self.da_total_windows = 0
        self.value_da.setText(
            "<span style='font-size:16px;color:#666;'>Domain Adaptation: warming up...</span>"
        )

        self.streaming = True
        self.shimmer_thread = ShimmerThread(PORT, self)
        self.shimmer_thread.start()
        self.last_predict_ts = 0.0
        self.setWindowTitle("ECG → BP Prediction (Streaming ON)")
        print("[INFO] Start streaming...")

    def stop_stream(self):
        if hasattr(self, "shimmer_thread"):
            self.shimmer_thread.running = False
        self.streaming = False
        self.setWindowTitle("ECG → BP Prediction (Streaming OFF)")
        print("[INFO] Stop streaming...")

    # ---------- Main Loop ----------
    def update_loop(self):
        if not self.streaming or len(self.buffer) < 50:
            return

        sig = np.array(self.buffer, float)

        # Pre-processing: de-mean + bandpass 0.5–40 Hz + notch 50 Hz
        sig = sig - np.mean(sig)

        try:
            b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
            sig_f = filtfilt(b, a, sig)

            bn, an = iirnotch(50.0 / (FS / 2), Q=30)
            sig_f = filtfilt(bn, an, sig_f)
        except Exception as e:
            print(f"[Filter ERROR] {e}")
            sig_f = sig

        # Plot ECG
        t = np.arange(len(sig_f)) / FS
        self.curve.setData(t, sig_f)

        if len(sig_f) > 0:
            amp = np.max(np.abs(sig_f))
            if amp <= 0:
                amp = 1.0
            self.plot.setYRange(-1.2 * amp, 1.2 * amp)

        # Detect R-peaks
        try:
            sig_norm = sig_f / (np.max(np.abs(sig_f)) + 1e-9)
            r_raw = self.detector.rpeak_detection(sig_norm, FS)
            r_peaks = refine_rpeaks(sig_norm, r_raw, FS)
        except Exception as e:
            print(f"[R-PEAK ERROR] {e}")
            r_peaks = np.array([], dtype=int)

        # update scatter
        pts = []
        for rp in r_peaks:
            if 0 <= rp < len(sig_f):
                pts.append({"pos": (rp / FS, sig_f[rp])})
        self.rpeak_scatter.setData(pts)
        self.value_rpeaks.setText(f"<span style='font-size:18px;'>R-peaks: {len(r_peaks)}</span>")

        # checking cukup panjang untuk window fitur
        enough_data = len(sig_f) >= int(FEATURE_WINDOW_SEC * FS)
        if not enough_data:
            return

        now = time.time()
        if self.last_predict_ts == 0 or (now - self.last_predict_ts) >= PREDICT_INTERVAL:
            self.last_predict_ts = now
            self._run_predict(sig_f, r_peaks)

    # ---------- Domain Adaptation Helper ----------
    def _update_da_stats(self, feat):
        """Update online mean/std fitur Shimmer."""
        for name, st in self.da_stats.items():
            if name not in feat:
                continue
            val = feat[name]
            if val is None or not math.isfinite(val):
                continue
            n = st["n"] + 1
            delta = val - st["mean"]
            mean = st["mean"] + delta / n
            M2 = st["M2"] + delta * (val - mean)
            st["n"], st["mean"], st["M2"] = n, mean, M2

        self.da_total_windows += 1

        # checking apakah semua fitur punya cukup sampel dan varian
        if not self.da_ready and self.da_total_windows >= DA_MIN_SAMPLES:
            ready = True
            for name, st in self.da_stats.items():
                if st["n"] < max(3, DA_MIN_SAMPLES // 2):
                    ready = False
                    break
                if st["n"] > 1:
                    std = math.sqrt(st["M2"] / (st["n"] - 1))
                    if std < 1e-6:
                        ready = False
                        break
            self.da_ready = ready
            if self.da_ready:
                print("[INFO] Domain Adaptation siap. Fitur Shimmer akan dipetakan ke domain training.")
                self.value_da.setText(
                    "<span style='font-size:16px;color:#2e7d32;'>"
                    "Domain Adaptation: ACTIVE</span>"
                )

    def _apply_domain_adapt(self, feat):
        """Map fitur Shimmer → domain training (TRAINING_STATS)."""
        if not self.da_ready:
            return feat  # belum siap, pakai apa adanya dulu

        adapted = dict(feat)  
        for name, target in TRAINING_STATS.items():
            if name not in feat:
                continue
            st = self.da_stats[name]
            if st["n"] < 2:
                continue
            mu_s = st["mean"]
            std_s = math.sqrt(st["M2"] / (st["n"] - 1))
            if std_s < 1e-6:
                continue

            mu_t = target["mean"]
            std_t = target["std"] if target["std"] > 0 else 1.0

            z = (feat[name] - mu_s) / std_s
            adapted[name] = z * std_t + mu_t

        return adapted

    # ---------- Prediction ----------
    def _run_predict(self, sig_f, r_peaks):
        win = int(FEATURE_WINDOW_SEC * FS)
        offset = len(sig_f) - win
        if offset < 0:
            return

        buf = sig_f[offset:]
        r_win = r_peaks[r_peaks >= offset] - offset
        if len(r_win) < 3:
            print(f"[WARN] R-peak terlalu sedikit dalam window (len={len(r_win)}), skip prediksi.")
            return

        feat = self.extractor.compute(buf, r_win)
        if feat is None:
            print("[WARN] Fitur tidak lengkap, skip prediksi.")
            return

        # update statistik Shimmer & lakukan domain adaptation
        self._update_da_stats(feat)
        feat_for_model = self._apply_domain_adapt(feat)

        sbp_mean, sbp_std, dbp_mean, dbp_std, err = self.predictor.predict(feat_for_model)
        if err or sbp_mean is None:
            print(f"[ERROR] Prediksi gagal: {err}")
            return

        # clipping ringan pada range fisiologis
        sbp_mean = float(np.clip(sbp_mean, 80, 200))
        dbp_mean = float(np.clip(dbp_mean, 40, 130))

        # smoothing
        self.sbp_hist.append(sbp_mean)
        self.dbp_hist.append(dbp_mean)
        self.hr_hist.append(feat.get("HR(bpm)", np.nan))
        self.hrv_hist.append(feat.get("HRV(ms)", np.nan))

        sbp_s = float(sbp_mean)
        dbp_s = float(dbp_mean)
        hr_s = float(np.nanmean(self.hr_hist))
        hrv_s = float(np.nanmean(self.hrv_hist))

        label, color = Category.label_and_color(sbp_s, dbp_s)

        # update label utama
        self.value_bp.setText(
            f"<span style='font-size:44px;font-weight:800;'>{sbp_s:.0f}/{dbp_s:.0f}</span> mmHg"
        )
        self.value_bp_std.setText(
            f"<span style='font-size:18px;'>± {sbp_std:.1f} / {dbp_std:.1f} mmHg</span>"
        )
        self.value_cat.setText(
            f"<span style='font-size:20px;color:rgb({color[0]},{color[1]},{color[2]});'>"
            f"Kategori: {label}</span>"
        )

        if np.isfinite(hr_s):
            self.value_hr.setText(
                f"<span style='font-size:20px;'>HR: {hr_s:.1f} bpm</span>"
            )
        else:
            self.value_hr.setText("<span style='font-size:20px;'>HR: - bpm</span>")

        if np.isfinite(hrv_s):
            self.value_hrv.setText(
                f"<span style='font-size:20px;'>HRV: {hrv_s:.1f} ms</span>"
            )
        else:
            self.value_hrv.setText("<span style='font-size:20px;'>HRV: - ms</span>")

        # log
        self._append_log(
            datetime.now().strftime("%H:%M:%S"),
            sbp_s, sbp_std,
            dbp_s, dbp_std,
            label, color,
            hr_s, hrv_s
        )

    # ---------- Log Table ----------
    def _append_log(self, waktu, sbp, sbp_std, dbp, dbp_std, label, color, hr, hrv):
        row = self.table.rowCount()
        self.table.insertRow(row)

        vals = [
            waktu,
            f"{sbp:.0f}",
            f"{sbp_std:.1f}",
            f"{dbp:.0f}",
            f"{dbp_std:.1f}",
            label,
            f"{hr:.1f}" if np.isfinite(hr) else "-",
            f"{hrv:.1f}" if np.isfinite(hrv) else "-",
        ]

        for i, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if i == 5:  # kolom kategori
                item.setForeground(QtGui.QBrush(QtGui.QColor(*color)))
            self.table.setItem(row, i, item)

        self.table.scrollToBottom()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    gui = ECGGui()
    gui.show()
    sys.exit(app.exec_())
