Continuous Prediction of Multiple Sclerosis Severity Using Environmental and Clinical Data

This repository supports the manuscript titled "The Effect of Environmental Exposure on Multiple Sclerosis Severity Score: A Study Based on Sequential Data Modeling", which investigates the progression of Multiple Sclerosis (MS) using deep learning techniques on integrated clinical and environmental data.
🔍 Overview

This project presents a deep learning framework to predict the Multiple Sclerosis Severity Score (MSSS) using:

    Longitudinal clinical data

    Demographic features

    Environmental exposure data (air pollution and weather conditions)

The approach combines temporal modeling, data imputation, and feature selection to produce a robust pipeline for disease progression prediction.
🧠 Key Contributions

    Data Integration: Clinical visits were enriched with environmental exposure data using inverse distance weighting (IDW) based on the closest weather/air quality stations.

    Hybrid Imputation: Missing values in longitudinal clinical records were imputed using a hybrid of Exponentially Weighted Moving Average (EWMA) and Linear Mixed Effects (LME) models.

    Feature Selection: Utilized AutoML (AutoGluon) to identify the most predictive clinical and environmental features.

    Deep Learning Models: Implemented RNN, LSTM, and GRU networks to forecast MSSS classes using 2–4 prior observations.

    Post-Hoc Interpretability: Applied SHAP to interpret model decisions and assess feature contributions.

📁 Repository Structure

.
├── data/
│   ├── raw/                  # Raw clinical and environmental datasets (anonymized)
│   ├── processed/            # Preprocessed and aligned datasets
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_selection_autogluon.ipynb
│   ├── 03_model_training_gru_lstm_rnn.ipynb
│   ├── 04_shap_interpretation.ipynb
│
├── src/
│   ├── imputation/
│   │   └── ewma_lme.py       # EWMA + LME hybrid imputation
│   ├── modeling/
│   │   ├── dataset.py        # Dataset creation and sequence batching
│   │   ├── models.py         # RNN, LSTM, GRU architectures
│   │   └── train.py          # Training loop with cross-validation
│   └── utils/
│       └── evaluation.py     # Evaluation metrics and plotting
│
├── results/
│   ├── model_outputs/
│   └── figures/
│
├── environment.yml           # Conda environment setup
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE

📊 Data Sources

    Clinical data: IRCCS Mondino Foundation, Pavia (2013–2022)

    Air quality data: European Environment Agency (PM2.5, PM10, NO₂, CO, O₃, SO₂)

    Weather data: E-OBS (temperature, humidity, radiation, precipitation, pressure, wind)

    ⚠️ Note: Due to privacy regulations, raw patient data is not shared publicly. Scripts are designed to work with a standardized anonymized format.

⚙️ Requirements

    Python 3.11

    R 4.2.3 (for imputation)

    AutoGluon

    PyTorch

    SHAP

    Pandas, NumPy, Scikit-learn

Install with:

conda env create -f environment.yml
conda activate ms-progression

📈 Performance Summary

    Best model: GRU with 4 prior visits + environmental data

    Average ROC-AUC: 0.827

    Clinical features (e.g., pyramidal, cerebellar scores) were top predictors

    Environmental features (PM2.5, NO₂, humidity, precipitation) improved predictions significantly

📝 Citation

If you use this code or data, please cite:
