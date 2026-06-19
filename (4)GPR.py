# ============================================================
#  GPR Revisi FINAL (V2)
#  - Kernel (RBF + Matern)
#  - Subset Record (50 pertama)
#  - Menyimpan:
#       * gpr_sbp_revisi.pkl
#       * gpr_dbp_revisi.pkl
#       * models/features_order.json         (urutan fitur final)
#       * models/feature_definition.json    (deskripsi fitur)
# ============================================================

import os
import warnings
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, WhiteKernel, ConstantKernel
)
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from joblib import dump

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI DASAR
# ============================================================
fs = 125
file_path = r"F:\Skripsi\Percobaan_Pantompkins\PT-Data-Baru\Data\part_1.mat"
rpeak_csv = "./results/Rpeak_all_records.csv"
feat_csv = "./results/ECG_features_summary.csv"
out_csv = "./results/BP_predictions_revisi.csv"  

os.makedirs("./results", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def moving_average(x, k=5):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(int(k)) / k, mode="same")

def winsorize(x, lo=0.01, hi=0.99):
    ql, qh = np.quantile(x, [lo, hi])
    return np.clip(x, ql, qh)

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

# ============================================================
# LOAD DATA
# ============================================================
print("[INFO] Loading MAT file:", file_path)
mat = loadmat(file_path)
main_key = [k for k in mat.keys() if not k.startswith("__")][0]
signal_all = mat[main_key]

print("[INFO] Loading R-peaks CSV:", rpeak_csv)
rpk = pd.read_csv(rpeak_csv)

print("[INFO] Loading ECG feature CSV:", feat_csv)
df_feat = pd.read_csv(feat_csv)

all_recs = sorted(rpk["Record_ID"].unique().astype(int))
chosen_recs = all_recs[:50]
print(f"[INFO] Memakai {len(chosen_recs)} record pertama dari total {len(all_recs)}")

# Pastikan tipe Record_ID konsisten
rpk["Record_ID"] = rpk["Record_ID"].astype(int)
df_feat["Record_ID"] = df_feat["Record_ID"].astype(int)

# ============================================================
# EKSTRAKSI LABEL SBP/DBP DARI ABP (per-beat, per-window)
# ============================================================
def estimate_bp_per_record(record_id, smooth_k=7, min_seg_s=0.18):
    """
    Mengestimasi SBP/DBP untuk tiap window (20 s, step 10 s) pada 1 record:
      - ABP dismoothing dengan moving average
      - Window 20 s, step 10 s
      - Ambil R-peak global record
      - Dalam tiap window, digunakan segmen antar R-peak:
           * DBP = min ABP(prev_r → r)
           * SBP = max ABP(r → next_r)
        lalu di-mean-kan.
    """
    rec = signal_all[0, record_id - 1]
    abp = moving_average(rec[1, :].astype(float), k=smooth_k)

    r_all = rpk.loc[rpk["Record_ID"] == record_id, "Index"].values.astype(int)
    r_all = r_all[(r_all > 0) & (r_all < len(abp) - 1)]
    if len(r_all) < 5:
        return None

    win_sec, step_sec = 20, 10
    W, S = win_sec * fs, step_sec * fs

    rows = []
    for start in range(0, len(abp) - W + 1, S):
        stop = start + W
        r_seg = r_all[(r_all >= start) & (r_all < stop)] - start
        if len(r_seg) < 3:
            continue

        abp_seg = abp[start:stop]
        min_len = int(min_seg_s * fs)

        sbp_vals, dbp_vals = [], []
        for i in range(1, len(r_seg) - 1):
            prev_r, r, nxt = r_seg[i - 1], r_seg[i], r_seg[i + 1]

            # DBP: trough sebelum R (prev_r → r)
            if r - prev_r >= min_len:
                dbp_vals.append(np.nanmin(abp_seg[prev_r:r]))

            # SBP: peak setelah R (r → next_r)
            if nxt - r >= min_len:
                sbp_vals.append(np.nanmax(abp_seg[r:nxt]))

        if not sbp_vals or not dbp_vals:
            continue

        sbp_val, dbp_val = np.nanmean(sbp_vals), np.nanmean(dbp_vals)

        # sanity check: SBP harus > DBP + 5
        if np.isnan(sbp_val) or np.isnan(dbp_val) or sbp_val <= dbp_val + 5:
            continue

        rows.append({
            "Record_ID": record_id,
            "Epoch_Start": start,
            "SBP_true": sbp_val,
            "DBP_true": dbp_val
        })

    return rows

print("[INFO] Mengestimasi SBP/DBP per window per record...")
bp_rows = []
for rid in chosen_recs:
    rec_rows = estimate_bp_per_record(rid)
    if rec_rows:
        bp_rows.extend(rec_rows)

df_bp = pd.DataFrame(bp_rows)
if df_bp.empty:
    raise SystemExit("[ERROR] Tidak ada label BP valid. Cek R-peak atau sinyal ABP.")

print(f"[INFO] Total window dengan label BP (sebelum filter): {len(df_bp)}")

# Winsorize label untuk stabilisasi
df_bp["SBP_true"] = winsorize(df_bp["SBP_true"].values)
df_bp["DBP_true"] = winsorize(df_bp["DBP_true"].values)

print("[INFO] Label SBP/DBP sudah di-winsorize (1% - 99%).")

# ============================================================
# TAMBAH FITUR HR DAN HRV DARI R-PEAK
# ============================================================
def calc_hr_features(rpeak_df, fs):
    rows = []
    for rid in rpeak_df["Record_ID"].unique():
        r_locs = rpeak_df.loc[rpeak_df["Record_ID"] == rid, "Index"].values
        rr = np.diff(r_locs) / fs
        if len(rr) < 3:
            hr, hrv = np.nan, np.nan
        else:
            hr = 60.0 / np.mean(rr)
            hrv = np.std(rr) * 1000.0  # ms
        rows.append({"Record_ID": rid, "HR(bpm)": hr, "HRV(ms)": hrv})
    return pd.DataFrame(rows)

print("[INFO] Menghitung HR & HRV per record...")
df_hr = calc_hr_features(rpk, fs)

df_feat = df_feat.merge(df_hr, on="Record_ID", how="left")

# ============================================================
# GABUNG FITUR DAN LABEL
# ============================================================
print("[INFO] Menggabungkan fitur ECG + label BP...")
data = (
    df_feat
    .merge(df_bp, on="Record_ID", how="inner")
    .dropna()
    .reset_index(drop=True)
)
groups = data["Record_ID"].values.astype(int)

print(f"[INFO] Total sampel gabungan: {len(data)} dari {len(np.unique(groups))} record")

# ============================================================
# FEATURE SELECTION (Mutual Information)
# ============================================================
feat_cols_all = [
    c for c in df_feat.columns
    if c not in ["Record_ID", "Total_Rpeaks"]
]

print("[INFO] Total kandidat fitur:", len(feat_cols_all))

X_all = np.nan_to_num(data[feat_cols_all].values)
y_sbp = data["SBP_true"].values
y_dbp = data["DBP_true"].values

print("[INFO] Menghitung Mutual Information untuk SBP...")
mi_sbp = mutual_info_regression(X_all, y_sbp, random_state=42)
print("[INFO] Menghitung Mutual Information untuk DBP...")
mi_dbp = mutual_info_regression(X_all, y_dbp, random_state=42)

top_idx_sbp = np.argsort(mi_sbp)[::-1][:8]
top_idx_dbp = np.argsort(mi_dbp)[::-1][:8]

feat_sbp = [feat_cols_all[i] for i in top_idx_sbp]
feat_dbp = [feat_cols_all[i] for i in top_idx_dbp]

feat_common = list(set(feat_sbp) | set(feat_dbp))

# Pastikan HR & HRV masuk
for essential in ["HR(bpm)", "HRV(ms)"]:
    if essential not in feat_common:
        feat_common.append(essential)

# Urutkan fitur supaya konsisten 
feat_common = list(feat_common)

X = np.nan_to_num(data[feat_common].values)

print(f"[INFO] Fitur dipakai ({len(feat_common)}):")
for f in feat_common:
    print("   -", f)

# ============================================================
# DEFINISI MODEL GPR
# ============================================================
def make_gpr(n_features: int):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * (
            RBF(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-3, 1e5)
            )
            + Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-3, 1e5),
                nu=2.5
            )
        )
        + WhiteKernel(noise_level=0.05)
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.02,
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=10
        ))
    ])

# ============================================================
# CROSS VALIDATION (GroupKFold per Record) + UQ (std)
# ============================================================
gpr_sbp = make_gpr(X.shape[1])
gpr_dbp = make_gpr(X.shape[1])

n_splits = min(5, len(np.unique(groups)))
gkf = GroupKFold(n_splits=n_splits)

pred_rows = []
fold_preds_sbp, fold_preds_dbp = [], []
m_sbp, m_dbp = [], []

print(f"[INFO] Running GroupKFold (n_splits={n_splits})...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for fold, (tr, te) in enumerate(gkf.split(X, groups=groups), 1):
        print(f"\n[Fold {fold}] Training...")
        gpr_sbp.fit(X[tr], y_sbp[tr])
        gpr_dbp.fit(X[tr], y_dbp[tr])

        # === TAMBAH UNCERTAINTY ===
        sbp_mean, sbp_std = gpr_sbp.predict(X[te], return_std=True)
        dbp_mean, dbp_std = gpr_dbp.predict(X[te], return_std=True)

        yts, ytd = y_sbp[te], y_dbp[te]

        # Simpan utk plot per-fold (pakai mean seperti semula)
        fold_preds_sbp.append((yts, sbp_mean))
        fold_preds_dbp.append((ytd, dbp_mean))

        ms = mape(yts, sbp_mean)
        md = mape(ytd, dbp_mean)
        m_sbp.append(ms)
        m_dbp.append(md)

        r2_s = r2_score(yts, sbp_mean)
        r2_d = r2_score(ytd, dbp_mean)

        print(f"[Fold {fold}] SBP MAPE={ms:.2f}% | R²={r2_s:.3f}")
        print(f"[Fold {fold}] DBP MAPE={md:.2f}% | R²={r2_d:.3f}")

        for i in range(len(yts)):
            pred_rows.append([
                data.iloc[te[i]]["Record_ID"],
                data.iloc[te[i]]["Epoch_Start"],
                yts[i], sbp_mean[i], sbp_std[i],
                ytd[i], dbp_mean[i], dbp_std[i]
            ])

# ============================================================
# SIMPAN HASIL KE CSV (MEAN + STD)
# ============================================================
out = pd.DataFrame(
    pred_rows,
    columns=[
        "Record_ID", "Epoch_Start",
        "SBP_true", "SBP_pred", "SBP_std",
        "DBP_true", "DBP_pred", "DBP_std"
    ]
)

out["SBP_MAPE(%)"] = np.abs(out["SBP_true"] - out["SBP_pred"]) / out["SBP_true"].replace(0, np.nan) * 100
out["DBP_MAPE(%)"] = np.abs(out["DBP_true"] - out["DBP_pred"]) / out["DBP_true"].replace(0, np.nan) * 100

mean_sbp_mape = out["SBP_MAPE(%)"].mean()
mean_dbp_mape = out["DBP_MAPE(%)"].mean()

out.loc[len(out)] = [
    "MEAN", "-",
    "-", "-", "-", "-", "-", "-",
    mean_sbp_mape, mean_dbp_mape
]

out.to_csv(out_csv, index=False)

print(f"\n[INFO] Disimpan ke {out_csv}")
print(f"[INFO] SBP mean MAPE={mean_sbp_mape:.2f}%")
print(f"[INFO] DBP mean MAPE={mean_dbp_mape:.2f}%")

# ============================================================
# PLOT PER FOLD (SBP & DBP) 
# ============================================================
def plot_perfold(fold_preds, label, color, filename):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, (y_true, y_pred) in enumerate(fold_preds[:5]):
        r2 = r2_score(y_true, y_pred)
        ax = axes[i]
        ax.scatter(y_true, y_pred, color=color, alpha=0.7, edgecolors="k")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--")
        ax.set_title(f"Fold {i+1}: $R^2$ = {r2:.2f}")
        ax.set_xlabel(f"True {label} (mmHg)")
        ax.set_ylabel(f"Predicted {label} (mmHg)")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"GPR Prediction Results - {label} (Per Fold)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"./results/{filename}"
    plt.savefig(out_path, dpi=400)
    plt.close()
    print(f"[INFO] Plot {label} per fold disimpan di {out_path}")

plot_perfold(fold_preds_sbp, "SBP", "steelblue", "Plot_GPR_SBP_perfold_revisi.png")
plot_perfold(fold_preds_dbp, "DBP", "seagreen", "Plot_GPR_DBP_perfold_revisi.png")

# ============================================================
# PLOT GLOBAL DENGAN ERROR BAR (mean ± std)
# ============================================================
plt.figure(figsize=(7, 6))
plt.errorbar(
    out["SBP_true"][:-1], out["SBP_pred"][:-1],
    yerr=out["SBP_std"][:-1],
    fmt='o', alpha=0.5, capsize=3, label="Pred (± std)"
)
mn, mx = out["SBP_true"][:-1].min(), out["SBP_true"][:-1].max()
plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal Fit")
plt.xlabel("SBP True (mmHg)")
plt.ylabel("SBP Predicted (mmHg)")
plt.title("SBP Prediction with Uncertainty")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./results/SBP_with_uncertainty.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.errorbar(
    out["DBP_true"][:-1], out["DBP_pred"][:-1],
    yerr=out["DBP_std"][:-1],
    fmt='o', alpha=0.5, capsize=3, label="Pred (± std)"
)
mn, mx = out["DBP_true"][:-1].min(), out["DBP_true"][:-1].max()
plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal Fit")
plt.xlabel("DBP True (mmHg)")
plt.ylabel("DBP Predicted (mmHg)")
plt.title("DBP Prediction with Uncertainty")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./results/DBP_with_uncertainty.png", dpi=300)
plt.close()

print("[INFO] Plot uncertainty disimpan di ./results/SBP_with_uncertainty.png & DBP_with_uncertainty.png")

# ============================================================
# MODEL FINAL + DEFINISI FITUR 
# ============================================================
print("\n[INFO] Training FINAL model pada semua data...")

gpr_sbp_final = make_gpr(X.shape[1])
gpr_dbp_final = make_gpr(X.shape[1])

gpr_sbp_final.fit(X, y_sbp)
gpr_dbp_final.fit(X, y_dbp)

# Simpan model dengan NAMA LAMA (sesuai keinginanmu)
dump(gpr_sbp_final, "./models/gpr_sbp_revisi.pkl")
dump(gpr_dbp_final, "./models/gpr_dbp_revisi.pkl")
print("[INFO] Model final disimpan di ./models/gpr_sbp_revisi.pkl & gpr_dbp_revisi.pkl")


features_order_path = "./models/features_order.json"
with open(features_order_path, "w") as f:
    json.dump(feat_common, f, indent=4)
print(f"[INFO] Urutan fitur disimpan di {features_order_path}")


feature_def = {
    "features_order": feat_common,
    "note": (
        "Fitur dalam urutan ini digunakan sebagai input untuk "
        "gpr_sbp_revisi.pkl dan gpr_dbp_revisi.pkl. "
        "Nilai fitur yang diberikan ke model HARUS dalam domain yang sama "
        "seperti yang dihitung di script ini (tanpa scaling tambahan)."
    ),
    "label_info": {
        "SBP_true": {
            "winsorize_quantile": [0.01, 0.99],
            "unit": "mmHg"
        },
        "DBP_true": {
            "winsorize_quantile": [0.01, 0.99],
            "unit": "mmHg"
        }
    }
}

feature_def_path = "./models/feature_definition.json"
with open(feature_def_path, "w") as f:
    json.dump(feature_def, f, indent=4)
print(f"[INFO] Definisi fitur disimpan di {feature_def_path}")

print("\nDONE — Training prediksi_revisi selesai.")