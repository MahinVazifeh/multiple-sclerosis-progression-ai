# 🧠 MSSS Progression Prediction with Clinical & Environmental Data

📊 **Dataset Source (Nature Scientific Data):**
["The BRAINTEASER Datasets: Clinical, Wearable and Environmental Data for ALS & MS Progression Modeling"](https://www.nature.com/articles/s41597-025-06095-1)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Research-success)
![Domain](https://img.shields.io/badge/Domain-Clinical%20AI-informational)
![Focus](https://img.shields.io/badge/Focus-Time%20Series%20Modeling-orange)
![Disease](https://img.shields.io/badge/Disease-Multiple%20Sclerosis-purple)
![Exposure](https://img.shields.io/badge/Exposure-Environmental%20Data-green)

---

## 🚀 Project Overview

This repository provides a **deep learning pipeline for predicting Multiple Sclerosis Severity Score (MSSS)** using longitudinal clinical data enriched with environmental exposure.

The framework integrates:

* 🏥 Clinical time-series data (patient visits)
* 👤 Demographic variables
* 🌍 Environmental exposure (air pollution & weather)

> 💡 **Goal:** Forecast disease severity at future visits using patient-level sequential modeling.

---

## 📄 Associated Publication

**["The Effect of Environmental Exposure on Multiple Sclerosis Severity Score: A Study Based on Sequential Data Modeling"](https://your-paper-link.com)**
Mahin Vazifehdan et al., 2026 (*International Journal of Medical Informatics*)

---

## 🧠 Why This Project Matters

Modeling MS progression is challenging due to:

* 📉 Missing and irregular clinical observations
* 🔄 Longitudinal dependencies across visits
* 🌍 External environmental influences

This pipeline addresses these challenges by:

* aligning multi-source temporal data
* applying robust preprocessing and imputation
* leveraging sequence-based deep learning models
* ensuring reproducibility and modularity

---

## ⚙️ Pipeline Highlights

* 🌍 **Environmental–Clinical Data Integration**
  Weekly alignment using Inverse Distance Weighting (IDW)

* 🧹 **Data Preprocessing & Cleaning**
  Structured filtering, normalization, and consistency checks

* 🧬 **Feature Selection**
  AutoML-based importance ranking using AutoGluon

* 🔁 **Sequence Modeling**
  RNN, LSTM, and GRU models using 2–4 historical observations

* 📊 **Outcome Modeling**
  MSSS classification into severity groups

* 🔍 **Model Explainability**
  SHAP analysis to identify key prediction drivers

---

## 📊 Key Results

* **Best Model:** GRU (4 time steps)
* **ROC-AUC:** 0.827 (cross-validated)

**Top Predictive Features**

* 🏥 Clinical

  * Pyramidal score
  * Sensitive score
  * Cerebellar score

* 🌍 Environmental

  * Mean PM2.5
  * NO₂
  * PM10 exceedance
  * Humidity
  * Precipitation

---

## 🧩 Methodology Overview

The pipeline follows a structured workflow:

1. **Data Preprocessing**

   * Filtering patients
   * Handling missing values
   * Temporal alignment

2. **Feature Engineering & Selection**

   * Clinical + environmental features
   * AutoML-based feature ranking

3. **Sequence Construction**

   * Patient-level time windows
   * Variable sequence lengths

4. **Model Training**

   * Stratified group cross-validation
   * Deep learning sequence models

5. **Evaluation & Interpretation**

   * ROC-AUC, precision, recall
   * SHAP-based explainability

---

## 📂 Repository Structure

```bash
.
├── config.py
├── data_preprocessing.py
├── feature_selection.py
├── sequence_builder.py
├── model_builder.py
├── training.py
├── evaluation.py
├── main.py
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

or (recommended):

```bash
conda env create -f environment.yml
conda activate msss-prediction
```

---

### Run the pipeline

```bash
python main.py
```

> This script runs the full pipeline: preprocessing, feature selection, training, and evaluation.

---

## ⚠️ Notes

* 🔒 Data is **not included** due to privacy constraints
* ⚠️ Feature selection must be performed on training data only (to avoid leakage)
* 🔁 Results may vary slightly due to stochastic deep learning training

---

## 📖 Citation

```bibtex
@article{vazifehdan2026msss,
  title={The Effect of Environmental Exposure on Multiple Sclerosis Severity Score},
  author={Vazifehdan, Mahin et al.},
  journal={International Journal of Medical Informatics},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Mahin Vazifehdan
