<!-- Top banner GIF (put your own in /media and update the path) -->
<p align="center">
  <img src="media/gait-auth-banner.gif" alt="Acceleration-Based User Authentication Banner" width="80%">
</p>

<h1 align="center">⚡ Acceleration-Based User Authentication</h1>
<h3 align="center">AI/ML Coursework</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Made_with-MATLAB-orange?logo=mathworks&style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Biometrics-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Supervised-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  <!-- Small looping GIF of signals / waves -->
  <img src="media/sensor-wave-loop.gif" alt="Sensor Wave Animation" width="260">
</p>

---

## 🌟 Project Snapshot

> **Goal:** Authenticate users from their **walking pattern** using smartwatch acceleration signals,  
> while exploring how **feature selection** and **optimisation** impact performance.

- 🎯 **Users:** 10  
- 📅 **Sessions:** F-Day & M-Day  
- 📡 **Sensors:** Smartwatch accelerometer (gyroscope reserved for future work)  
- 🧠 **Models:** Feedforward Neural Networks, SVM + GA optimisation  
- 🧪 **Analysis:** PCA, Cosine similarity, DTW, Inter/Intra variance, ratio splits  

---

## 📁 Repository Structure

```text
dataset/                         # MAT feature files (FDay / MDay)
dataset_csv/                     # Raw smartwatch CSV recordings

ANova_figures/                   # ANOVA + MI + SG feature-selection plots
SVMFigures/                      # SVM optimisation plots & logs

preprocess_csvs.m                # CSV → MAT feature pipeline
run_benchmark.m                  # Main NN benchmark (all scenarios)
initial_model.m                  # Baseline neural network
initial_model_tuning.m           # Hyperparameter tuning for NN

optimization_SVM_of_model.m      # GA + SVM optimisation (with LOUO)
optimization_SVM_no_louo_of_model.m

modelratiotester.m               # Target:imposter ratio experiments (1:1–1:7)

cluster_3d.m                     # PCA clustering visualisations
cosine_3d.m                      # Cosine similarity across days
user_similarity_dtw.m            # DTW similarity heatmaps
inter_andIntra_3d.m              # Inter vs intra-user variance analysis

testing_scenarios_results_clean.mat
benchmark_results*.mat
ratio_splitting_performance_*.mat
feature_analysis_results*.mat
visualizer_of_variance.m
````

---

## 🧩 Pipeline Overview

<p align="center">
  <!-- Architecture / pipeline GIF (PCA, NN, SVM animation etc.) -->
  <img src="media/pipeline-flow.gif" alt="Pipeline Animation" width="75%">
</p>

1. **Data Ingestion** → Raw CSVs from smartwatch
2. **Preprocessing** → Feature extraction (time + frequency)
3. **Feature Selection** → ANOVA + MI + Steepest Gradient
4. **Model Training** → NN per user (target vs imposters), SVM + GA
5. **Evaluation** → FAR, FRR, EER, MCC, ROC, confusion matrices
6. **Analysis** → PCA clusters, DTW, cosine similarity, variance plots

---

## ⚙️ How to Run

### 1️⃣ Optional: Rebuild MAT Features

```matlab
preprocess_csvs
```

* Reads `dataset_csv/`
* Extracts time & frequency features
* Writes feature sets into `dataset/`

---

### 2️⃣ Main NN Benchmark

```matlab
run_benchmark
```

Runs multiple scenarios and saves:

* Accuracy, FAR, FRR, EER, MCC
* Confusion matrices & ROC curves
* `benchmark_results*.mat`, `testing_scenarios_results_clean.mat`

---

### 3️⃣ Hybrid Feature Selection (ANOVA + MI + SG)

```matlab
ANOVa_steepest_gradiant
```

Outputs:

* Stacked bar charts of feature importance
* Correlation matrices of selected subsets
* Top-k feature rankings
* Figures under `ANova_figures/`

---

### 4️⃣ Visual Insights (PCA, DTW, Similarity)

```matlab
cluster_3d          % PCA clusters
cosine_3d           % F-Day vs M-Day cosine similarity
user_similarity_dtw % DTW heatmaps
inter_andIntra_3d   % Inter vs intra-user variance
```

<p align="center">
  <!-- PCA rotation GIF -->
  <img src="media/pca-rotation.gif" alt="PCA 3D Cluster Rotation" width="55%">
</p>

---

### 5️⃣ SVM + GA Optimisation

```matlab
optimization_SVM_of_model        % Full GA + LOUO evaluation
% or (faster, without leave-one-user-out)
optimization_SVM_no_louo_of_model
```

Generates:

* Optimised SVM hyperparameters
* Selected feature subsets
* Performance summaries & figures → `SVMFigures/`

---

## 📊 Key Experiments & Metrics

* 🔁 **Target:Imposter Ratios:** `modelratiotester` explores 1:1 … 1:7
* 📉 **Metrics:** Accuracy, FAR, FRR, EER, Precision, Recall, MCC, AUC
* 🧬 **Feature Sets:** TimeD, FreqD, Combined (`Acc_TimeD_FreqD`)
* 🧪 **Generalisation:** Train on F-Day, test on M-Day & combined-day setups

---

## 🧑‍💻 Contributors

<p align="center">
  <!-- subtle fade / hover comes from GitHub’s default styles; layout is the “cool” part -->
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/Sithumsankajith">
          <img src="https://avatars.githubusercontent.com/Sithumsankajith" width="110" style="border-radius:50%;"><br>
          <sub><b>Sithum Sankajith</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/dasunikayapabandara">
          <img src="https://avatars.githubusercontent.com/dasunikayapabandara" width="110" style="border-radius:50%;"><br>
          <sub><b>Dasunika Yapabandara</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/BenoliSenanayake">
          <img src="https://avatars.githubusercontent.com/BenoliSenanayake" width="110" style="border-radius:50%;"><br>
          <sub><b>Benoli Senanayake</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/DinuriJayaweera">
          <img src="https://avatars.githubusercontent.com/DinuriJayaweera" width="110" style="border-radius:50%;"><br>
          <sub><b>Dinuri Jayaweera</b></sub>
        </a>
      </td>
    </tr>
  </table>
</p>

---

## 📜 License & Usage

This repository is created for **AI/ML coursework** and is **not intended for commercial use**.


---

````

If you want, tell me **which figures you already have** (PCA, ROC, DTW, etc.), and I’ll suggest exactly **which ones to convert to GIF** and where to place them in the README.
