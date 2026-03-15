# MSSS Progression Prediction with Clinical and Environmental Data

This repository contains code, models, and preprocessing workflows supporting the manuscript:

**"The Effect of Environmental Exposure on Multiple Sclerosis Severity Score: A Study Based on Sequential Data Modeling"**  
Mahin Vazifehdan et al., 2026 (Published in *International Journal of Medical Informatics*)

---

## 📌 Project Summary

We developed a deep learning framework to predict **Multiple Sclerosis Severity Score (MSSS)** classes by leveraging:

- Longitudinal clinical records  
- Demographic variables  
- Environmental exposures (air pollution and weather conditions)

Our models integrate clinical history and environmental trends to forecast disease severity at future visits using patient-level time series data.

---

## 🔍 Main Features

- **Data Integration**: Weekly-aligned environmental exposures joined to patient visits using Inverse Distance Weighting (IDW).
- **Imputation**: Hybrid imputation strategy combining EWMA and Linear Mixed Effects (LME) models.
- **Feature Selection**: AutoML-based feature importance via AutoGluon.
- **Deep Learning Models**: GRU, LSTM, and RNN architectures trained on 2–4 prior visits.
- **Interpretability**: SHAP analysis to explain prediction drivers.

---

## 🧠 Model Performance

- **Best Model**: GRU with 4 prior observations
- **ROC-AUC**: 0.827 (average across folds)
- **Top Features**:
  - Clinical: Pyramidal, Sensitive, Cerebellar scores
  - Environmental: Mean PM2.5, NO₂, PM10 exceedance, Humidity, Precipitation

---

## 📁 Repository Structure

```bash
.
├── data/                     # Folder for raw and processed datasets (not included here)
├── notebooks/                # Jupyter notebooks for each pipeline step
│   ├── preprocessing.ipynb
│   ├── autogluon_features.ipynb
│   ├── model_training.ipynb
│   └── shap_analysis.ipynb
├── src/                      # Source code
│   ├── imputation/
│   ├── modeling/
│   ├── utils/
├── results/                  # Results and plots
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
