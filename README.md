# 📌 Acceleration-Based User Authentication

### *AI/ML Coursework*

This repository contains the full MATLAB implementation, experimental workflow, feature analysis, and visualisations for an **acceleration-based user authentication system** developed using smartwatch motion data.
The project investigates whether **walking patterns** captured from inertial sensors can reliably authenticate users under realistic conditions.

---

# 🚀 1. Project Overview

* **Device:** Smartwatch inertial sensors
* **Sensors:** Accelerometer (used), Gyroscope (available but reserved for future work)
* **Users:** 10 participants
* **Sessions:**

  * **F-Day** (Session 1)
  * **M-Day** (Session 2 – collected separately)
* **Feature Sets:**

  * Time Domain (`Acc_TimeD`)
  * Frequency Domain (`Acc_FreqD`)
  * Combined (`Acc_TimeD_FreqD`)
* **Core Tasks:**

  * Train neural network & SVM-based authentication models
  * Analyse similarity across sessions
  * Perform hybrid feature selection (ANOVA + MI + SG)
  * Optimise model performance (GA + SVM tuning, NN tuning)
  * Generate PCA clusters, DTW matrices, variance plots
  * Evaluate real-world authentication metrics (Accuracy, FAR, FRR, EER, MCC)

---

# 📁 2. Repository Structure

```text
dataset/                         # MAT feature files per user (F-Day / M-Day)
dataset_csv/                     # Raw CSV recordings from smartwatch

ANova_figures/                   # ANOVA + MI + SG feature selection figures
SVMFigures/                      # SVM optimisation results & figures

preprocess_csvs.m                # CSV → MAT feature extraction and preprocessing
run_benchmark.m                  # Main NN benchmark + evaluation scenarios
initial_model.m                  # Baseline neural network
initial_model_tuning.m           # Hyperparameter tuning for NN

optimization_SVM_of_model.m      # Full GA + SVM optimisation (LOU enabled)
optimization_SVM_no_louo_of_model.m
                                 # Fast SVM optimisation (no leave-one-user-out)
modelratiotester.m               # Tests different target:imposter ratios (1:1–1:7)

ANOVa_steepest_gradiant.m        # Hybrid feature selection (ANOVA + MI + SG)
ANOVa_steepest_gradiant_without_lou.m

cluster_3d.m                     # PCA clustering in 2D/3D
cosine_3d.m                      # Cosine similarity (F-Day vs M-Day)
user_similarity_dtw.m            # DTW-based inter-user similarity matrix
inter_andIntra_3d.m              # Inter-user vs intra-user variance analysis

feature_set_split_tester_of_model.m
                                 # Feature-set-wise performance comparison
testing_scenarios_results_clean.mat
benchmark_results*.mat           # Summary metrics for all models
ratio_splitting_performance_*.mat
feature_analysis_results*.mat
visualizer_of_variance.m
```

---

# ▶️ 3. Running the Experiments

## 3.1. Requirements

* **MATLAB R2023a+** recommended
* **Toolboxes:**

  * Statistics and Machine Learning Toolbox
  * Neural Network Toolbox

Clone the repo and open MATLAB in the root directory.

---

## 3.2. Preprocessing (Optional)

```matlab
preprocess_csvs
```

This script:
✓ Reads raw CSVs
✓ Extracts time & frequency features
✓ Creates the MAT feature files:

* `Acc_TimeD_FDay`, `Acc_TimeD_MDay`
* `Acc_FreqD_FDay`, `Acc_FreqD_MDay`
* `Acc_TimeD_FreqD_FDay`, `Acc_TimeD_FreqD_MDay`

---

## 3.3. Neural Network Benchmark

```matlab
run_benchmark
```

Outputs include:

* Confusion matrices
* ROC curves
* FAR / FRR / EER
* MCC
* Ratio variations
* Scenario-based summaries
* `benchmark_results.mat`

---

## 3.4. Feature Selection (ANOVA + MI + SG)

```matlab
ANOVa_steepest_gradiant
```

Produces:

* Feature importance bar charts
* MI & p-value visualisations
* Correlation heatmaps
* Top-k feature rankings
* Saved under **ANova_figures/**

---

## 3.5. PCA Clustering

```matlab
cluster_3d
```

Creates user-labelled PCA scatterplots in 2D & 3D.
Ideal for showing class separability.

---

## 3.6. Ratio Splitting (Target:Imposter)

```matlab
modelratiotester
```

Generates:

* Accuracy vs Ratio
* FAR / FRR / EER curves
* MCC behaviour
* Saved as `ratio_splitting_performance_*.mat`

---

## 3.7. SVM + GA Optimisation

Full optimisation:

```matlab
optimization_SVM_of_model
```

Fast version:

```matlab
optimization_SVM_no_louo_of_model
```

Produces:

* Best chromosomes
* Selected feature sets
* Optimised hyperparameters
* SVM performance figures

---

# 🧪 4. Supporting Analytical Tools

### ✔ Cosine Similarity (`cosine_3d.m`)

Evaluates session-to-session consistency.

### ✔ Dynamic Time Warping (`user_similarity_dtw.m`)

Shows similarity structure between users.

### ✔ Inter vs Intra Variance (`inter_andIntra_3d.m`)

Checks biometric discriminability.

---

# 🔄 5. Recommended Workflow

1. *(Optional)* Rebuild MAT files → `preprocess_csvs`
2. Train & benchmark NN models → `run_benchmark`
3. Perform feature selection → `ANOVa_steepest_gradiant`
4. Analyse clusters → `cluster_3d`
5. Similarity analysis → `cosine_3d`, `user_similarity_dtw`
6. Ratio analysis → `modelratiotester`
7. SVM optimisation → `optimization_SVM_of_model`
8. Use generated MAT files & figures for the report

---

# 👥 Contributors

<div align="center">

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Sithumsankajith">
        <img src="https://avatars.githubusercontent.com/Sithumsankajith" width="120" style="border-radius:50%;"><br>
        <sub><b>Sithum Sankajith</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/dasunikayapabandara">
        <img src="https://avatars.githubusercontent.com/dasunikayapabandara" width="120" style="border-radius:50%;"><br>
        <sub><b>Dasunika Yapabandara</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/BenoliSenanayake">
        <img src="https://avatars.githubusercontent.com/BenoliSenanayake" width="120" style="border-radius:50%;"><br>
        <sub><b>Benoli Senanayake</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/DinuriJayaweera">
        <img src="https://avatars.githubusercontent.com/DinuriJayaweera" width="120" style="border-radius:50%;"><br>
        <sub><b>Dinuri Jayaweera</b></sub>
      </a>
    </td>
  </tr>
</table>

</div>
