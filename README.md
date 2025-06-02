# MSSS Classification (Flexible Sequence Modeling)

This repository contains a deep learning pipeline to forecast the class of MSSS score for MS (Multiple Sclerosis) patients. The model supports **flexible sequence lengths**, enabling temporal modeling of patient observation histories using **RNN, LSTM,** and **GRU** architectures.

---

## 🔍 Overview

The MSSS classification groups used in this project are based on predefined deciles:

- **Group 1 (Low)**: MSSS classified value = 1
- **Group 2 (Mild)**: MSSS classified value = 2 or 3
- **Group 3 (Moderate/High)**: MSSS classified value ≥ 4

The goal is to **predict the next MSSS group** based on sequences of clinical, demographic, and environmental features over time.

---

## 🧠 Features

- Sequence-based classification using **RNN / LSTM / GRU**
- Customizable **sequence length** for temporal input
- **MinMaxScaler** normalization and patient-wise time-series grouping
- Stratified group k-fold validation
- **AutoGluon** for feature selection (optional)
- SHAP explainability on deep models
- Flexible model training and evaluation pipeline
