# Cleaned version: MSSS Classification (Group 3 only)

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import shap
import pickle

# Load and filter dataset
data = pd.read_csv("Clinical_Demographic_Treatment_AllFeatures.csv")
data = data[data.num_obs < 40]
data = data[~data.MSSS_classified.isnull()]
data['Patient_number'] = data['Patient_ID'].factorize()[0] + 1
data["observation_time"] = data.groupby("Patient_number").cumcount() + 1

# MSSS Group 3 classification
# Low: 1, Mild: 2-3, Moderate-High: 4-6
data["MSSS_Classify"] = data.apply(
    lambda row: 1 if row["MSSS_classified"] == 1
    else 2 if row["MSSS_classified"] in [2, 3]
    else 3,
    axis=1
)

# Drop columns
columns_to_remove = np.r_[1:6, 7:14]  # Adjust based on actual columns for Clinical+Env+Demog
data.drop(["MSSS", "DiseaseDuration_Years", "EDSS"], axis=1, inplace=True)
data.drop(data.columns[columns_to_remove], axis=1, inplace=True)

# Train/test split
patients = data['Patient_number'].unique().tolist()
train_index, test_index = train_test_split(patients, test_size=0.1, random_state=1)
train_data = data[data['Patient_number'].isin(train_index)]
test_data = data[data['Patient_number'].isin(test_index)]

# Normalize
def normalize_data(train_data, test_data, columns_to_select):
    scaler = MinMaxScaler()
    norm_train = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    norm_test = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
    norm_train[columns_to_select] = train_data[columns_to_select].reset_index(drop=True)
    norm_test[columns_to_select] = test_data[columns_to_select].reset_index(drop=True)
    norm_train.drop(columns=['Patient_number', 'observation_time'], inplace=True)
    norm_test.drop(columns=['Patient_number', 'observation_time'], inplace=True)
    return norm_train, norm_test

columns_to_select = ['Patient_number', 'observation_time', 'MSSS_Classify']
normalized_train, normalized_test = normalize_data(train_data, test_data, columns_to_select)
# Define model

def keras_experiment(seq_length, feature_size: int, model_name: str):
    if model_name == 'RNN':
        model_1 = SimpleRNN(16, return_sequences=True, input_shape=(seq_length, feature_size))
        model_2 = SimpleRNN(16)
    elif model_name == 'LSTM':
        model_1 = LSTM(16, return_sequences=True, input_shape=(seq_length, feature_size))
        model_2 = LSTM(16)
    elif model_name == 'GRU':
        model_1 = GRU(16, return_sequences=True, input_shape=(seq_length, feature_size))
        model_2 = GRU(16)
    else:
        raise ValueError(f'Model Name: {model_name} is Not Valid!')

    model = Sequential([
        model_1,
        Dropout(0.1),
        model_2,
        Dropout(0.1),
        Dense(32, activation='relu', kernel_regularizer=l2(0.8)),
        Dense(3, activation='softmax')
    ])
    model.compile(
        loss='CategoricalCrossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

# Sequence preprocessing (next observation)
def preprocess_multi(data, sequence_len: int):
    x, y = [], []
    for pid in data['Patient_number'].unique():
        pdata = data[data['Patient_number'] == pid].sort_values(by='observation_time').reset_index(drop=True)
        y_seq = pdata['MSSS_Classify'].values
        pdata = pdata.drop(['MSSS_Classify', 'Patient_number'], axis=1)

        if len(pdata) > sequence_len:
            for i in range(len(pdata) - sequence_len):
                x.append(pdata.iloc[i:i+sequence_len].values)
                y.append(y_seq[i + sequence_len])
        else:
            padded = pd.concat([pdata, pdata.tail(sequence_len - len(pdata) + 1)])
            padded = padded.iloc[:sequence_len + 1]
            y_seq = y_seq[:sequence_len + 1]
            x.append(padded.iloc[:sequence_len].values)
            y.append(y_seq[sequence_len])
    return np.array(x), np.array(y)

# Example: One run (can be looped as needed)
model_name = 'GRU'
seq_length = 3
train_x, train_y = preprocess_multi(normalized_train.assign(Patient_number=train_data['Patient_number'].values,
                                                            MSSS_Classify=train_data['MSSS_Classify'].values,
                                                            observation_time=train_data['observation_time'].values),
                                    sequence_len=seq_length)
train_y_onehot = tf.keras.utils.to_categorical(train_y - 1, num_classes=3)

model = keras_experiment(seq_length, train_x.shape[2], model_name)
history = model.fit(train_x, train_y_onehot, epochs=10, batch_size=32, validation_split=0.2)

# SHAP analysis
background = train_x[np.random.choice(train_x.shape[0], 100, replace=False)]
test_x, _ = preprocess_multi(normalized_test.assign(Patient_number=test_data['Patient_number'].values,
                                                    MSSS_Classify=test_data['MSSS_Classify'].values,
                                                    observation_time=test_data['observation_time'].values),
                             sequence_len=seq_length)
test_samples = test_x[:10]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_samples)
shap.image_plot(shap_values, test_samples)

# Plot loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title(f"{model_name} - Loss (Seq: {seq_length})")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_loss.jpg", dpi=300)

# Save model
model.save("model_group3.h5")
with open("train_data_group3.pickle", "wb") as f:
    pickle.dump(train_x, f)
