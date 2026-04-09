import tensorflow as tf
from tensorflow.keras.regularizers import l2

from config import LEARNING_RATE


def build_model(
    sequence_length: int,
    feature_size: int,
    model_name: str,
) -> tf.keras.Model:
    """
    Build and compile a sequence classification model.

    Parameters
    ----------
    sequence_length : int
        Number of time steps in each sequence.
    feature_size : int
        Number of features per time step.
    model_name : str
        One of {"RNN", "LSTM", "GRU"}.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """

    if model_name == "RNN":
        first_layer = tf.keras.layers.SimpleRNN(
            16,
            return_sequences=True,
            input_shape=(sequence_length, feature_size),
        )
        second_layer = tf.keras.layers.SimpleRNN(16)

    elif model_name == "LSTM":
        first_layer = tf.keras.layers.LSTM(
            16,
            return_sequences=True,
            input_shape=(sequence_length, feature_size),
        )
        second_layer = tf.keras.layers.LSTM(16)

    elif model_name == "GRU":
        first_layer = tf.keras.layers.GRU(
            16,
            return_sequences=True,
            input_shape=(sequence_length, feature_size),
        )
        second_layer = tf.keras.layers.GRU(16)

    else:
        raise ValueError(f"Invalid model name: {model_name}. Choose from ['RNN', 'LSTM', 'GRU'].")

    model = tf.keras.Sequential(
        [
            first_layer,
            tf.keras.layers.Dropout(0.1),
            second_layer,
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=l2(0.8),
            ),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        loss="categorical_crossentropy",  # cleaner string format
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model