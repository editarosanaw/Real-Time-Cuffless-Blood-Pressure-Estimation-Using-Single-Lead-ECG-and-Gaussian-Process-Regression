This repository provides the source code, configuration files, and anonymized processed data used in the study:

**“Real-Time Cuffless Blood Pressure Estimation Using Single-Lead ECG and Gaussian Process Regression”**

The study proposes a feasibility-oriented real-time framework for cuffless blood pressure estimation from wearable single-lead electrocardiography (ECG). The framework integrates Pan-Tompkins++ R-peak detection, ECG-derived feature extraction, lightweight feature-level domain adaptation, and Gaussian Process Regression (GPR) for systolic blood pressure (SBP) and diastolic blood pressure (DBP) prediction.

---

## 📌 Repository Overview

This repository includes:

* Pan-Tompkins++ ECG pre-processing and R-peak detection code
* ECG-derived feature extraction code
* Lightweight feature-level domain adaptation code
* GPR training and inference scripts
* Hyperparameter settings
* Anonymized processed ECG-derived feature data
* Anonymized 60-sample comparison table used in the manuscript

The raw wearable ECG recordings are not publicly released because the original participant consent and ethical approval did not include open sharing of identifiable or potentially re-identifiable physiological waveform data.

---

## 🎯 Research Purpose

This repository supports research on:

**Real-time cuffless blood pressure estimation using single-lead wearable ECG**

The main objective is to evaluate the feasibility of estimating SBP and DBP from ECG-derived morphological, temporal, and statistical features using Gaussian Process Regression under a controlled healthy-adult wearable testing setting.

The proposed framework is intended for feasibility-oriented window-level BP estimation and should not be interpreted as clinical validation of continuous beat-to-beat blood pressure monitoring.

---

## 🧠 Proposed Framework

The proposed framework consists of the following stages:

1. ECG pre-processing and R-peak detection using Pan-Tompkins++
2. ECG-derived feature extraction
3. Lightweight feature-level domain adaptation
4. GPR-based SBP and DBP estimation
5. Real-time sliding-window inference

The ECG-derived features include HR, HRV, QT interval, QTc interval, TQ interval, SDI, SDIn, RMS, Hjorth Mobility, and Hjorth Complexity.

---

## 📊 Data Description

The model was developed using a public cuffless blood pressure dataset containing ECG, PPG, and arterial blood pressure signals. In this study, only the ECG channel was used as model input, while arterial blood pressure was used to derive reference SBP and DBP labels.

For independent wearable testing, ECG signals were collected from 20 healthy adult volunteers using a Shimmer single-lead ECG device. Each participant completed three measurement sessions, producing 60 wearable testing samples. Reference SBP and DBP values were obtained using an Omron HEM-7120 digital sphygmomanometer validated according to the European Society of Hypertension International Protocol revision 2010.

This repository provides anonymized processed ECG-derived features, reference BP values, and predicted BP values. Raw wearable ECG recordings are not publicly released due to participant consent and ethical restrictions.

---

## ⚙️ Hyperparameter Settings

The hyperparameter configuration includes:

* Sampling rate
* Analysis window duration
* Prediction update interval
* Band-pass filter settings
* Pan-Tompkins++ parameters
* Feature extraction settings
* Domain adaptation settings
* GPR kernel configuration
* Random Forest baseline settings

---

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/[username]/single-lead-ecg-cuffless-bp-gpr.git
cd single-lead-ecg-cuffless-bp-gpr
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ECG-based BP estimation pipeline

```bash
python code/infer_gpr.py
```

---

## 🔒 Data Availability and Ethical Restriction

The raw wearable ECG recordings are not publicly available because the original participant consent and ethical approval did not include open sharing of identifiable or potentially re-identifiable physiological waveform data.

To support reproducibility, this repository provides:

* Source code for the proposed computational pipeline
* Hyperparameter settings
* Processed ECG-derived features
* Predicted SBP and DBP values

These files allow independent verification of the proposed computational workflow without releasing raw physiological waveform data.

---

## 📜 License

This repository is intended for academic and research purposes.

The source code is released under the MIT License, unless otherwise stated. The processed data are provided for non-commercial academic research and reproducibility purposes only.

Please cite the related publication when using this repository.

---

## 📌 Citation

If you use this repository, please cite:

```text
Widasari, E. R., Alhamid, F. K., Laksono, R. M., and Syauqi, D.
“Real-Time Cuffless Blood Pressure Estimation Using Single-Lead ECG and Gaussian Process Regression.”
International Journal of Intelligent Engineering and Systems, 2026.
```

---

## 👥 Contributors

* **Dr. Edita Rosana Widasari, S.T., M.T., M.Eng., Ph.D.**
  Faculty of Computer Science, Universitas Brawijaya, Indonesia

* **Faticha Kamila Alhamid**
  Faculty of Computer Science, Universitas Brawijaya, Indonesia

* **Ristiawan Muji Laksono**
  Faculty of Medicine, Universitas Brawijaya, Indonesia

* **Dahnial Syauqi**
  Faculty of Computer Science, Universitas Brawijaya, Indonesia

---

## 📬 Contact

For questions regarding this repository, please contact:

**Edita Rosana Widasari**
Faculty of Computer Science, Universitas Brawijaya
Email: [editarosanaw@ub.ac.id](mailto:editarosanaw@ub.ac.id)
