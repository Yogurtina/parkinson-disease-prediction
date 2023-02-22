# Feed Forward Neural Network
# I used batch normalisation.
# Instead of normalizing only once before applying the neural network,
# the output of each level is normalized and used as input of the next level.
# This speeds up the convergence of the training process.

from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import kerastuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import tensorflow as tf

import utils.common as common

# Data preperation

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data = pd.read_csv('data.csv', delimiter = ',', header=0)
data = data.drop(labels="name", axis=1)

labels = data['status'].to_numpy()
data = data.drop(labels="status", axis=1)

features = data[data.columns.to_numpy()]
features = min_max_scaler.fit_transform(features)

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)

print('Training records:',Y_train.size)
print('Test records:',Y_test.size)

input_shape = [X_train.shape[1]]

MODEL_FILE = "models/feedforward.model"

# Custom Model (No hyperparameter tuning)

def build_feedforward_network():

    early_stopping = keras.callbacks.EarlyStopping(
        patience=20,
        min_delta=0.001,
        restore_best_weights=True,
    )

    model = keras.Sequential([
        layers.BatchNormalization(),
        layers.Dense(20,activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dense(50, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=20,
        min_delta=0.001,
        restore_best_weights=True,
    )
  
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=30,
        epochs=100,
        callbacks=[early_stopping],
    )

    return model, history

# check if the model is saved in the current directory
# if not, train the model
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print('Model loaded')
except:
    print('Model not found, training model')
    model, history = build_feedforward_network()

    history = pd.DataFrame(history.history)
    history.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
    history.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

    model.save(MODEL_FILE)
    print('Model saved')

# Hyperparameter Tuning

def create_model_feedforward(hyperparameters):
    
    model = keras.Sequential()
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(hyperparameters.Int("units_input",1, 50, 10), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparameters.Float(f'dropout_input', 0, 0.5, 0.1)))

    # Tune number of layers

    for  i in range(0, hyperparameters.Int('num_layers', 1, 4, 1)):
        model.add(layers.Dense(hyperparameters.Int(f'units_{i}', 1, 50, 10), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hyperparameters.Float(f'dropout_{i}', 0, 0.5, 0.1)))

    # Output layer

    model.add(layers.Dense(1, activation='softmax'))
    
    # Learning rate

    learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
    )

    return model

tuner = kt.Hyperband(create_model_feedforward,
                     objective='val_binary_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='tuned_models',
                     project_name='FFNN')


stop_early_tuner = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early_tuner])
tuner.search_space_summary()

# Get the optimal hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hyperparameters)

early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

for metric in model.metrics_names:
    print(f"{metric}: {metrics_mean[metric]:.3f} +/- {metrics_std[metric]:.3f}")

def model_func(model, group_X_train, group_Y_train, group_X_test, group_Y_test):
    return model.fit(group_X_train, group_Y_train,
                        validation_data=(group_X_test, group_Y_test),
                        batch_size=30,
                        epochs=100,
                        verbose=0,
                        callbacks=[early_stopping],
                        )

common.check_model(model, model_func, "FeedForward Network")
tuner.results_summary()
