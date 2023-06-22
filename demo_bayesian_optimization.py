# pylint: disable=E1101
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras_tuner as kt

# Load the digits dataset
digits = datasets.load_digits()

# Split the dataset into features and labels
x = digits.images
y = digits.target

# Reshape the features to be flat, as expected by the Dense layer
x = x.reshape((x.shape[0], -1))

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


def build_model(hp):
    model = keras.Sequential(
        [
            keras.layers.Dense(
                hp.Int("first_hidden", 32, 256, step=32), activation="relu"
            ),
            keras.layers.Dense(
                hp.Int("second_hidden", 32, 256, step=32), activation="relu"
            ),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 0.005, 0.01, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = kt.BayesianOptimization(build_model, objective="val_accuracy", max_trials=10)

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

best_hyperparameters = tuner.get_best_hyperparameters()[0]

best_model = tuner.hypermodel.build(best_hyperparameters)
