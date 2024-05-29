from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.backend import sign
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import gc
import os
from time import sleep
from keras.models import model_from_json
from DataLoader import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
# we should specify shape of the input tensor
# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
Data = data_loader.preprocess_data()
X = Data.iloc[:,:-1]
y = Data.iloc[:,-1]



# K-fold Cross Validation model evaluation

current_fold = 1
kfold = StratifiedKFold(n_splits=10, shuffle=True)
list_acc = []
list_loss = []
list_history = []
for train, test in kfold.split(X, y):
        print("### Train on Fold: ", current_fold)
           # LSTM model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=32, output_dim=32, input_length=20))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=16, activation='relu'))
        model.add(Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        #microbatch must be divisible by units
        model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
        adam_modified = optimizers.Adam(learning_rate=0.001, beta_1=0.7, beta_2=0.9, amsgrad=False)

        model.compile(loss="categorical_crossentropy", optimizer=adam_modified, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor="val_loss",
                                                     mode="min", patience=25,
                                                     restore_best_weights=True)
        history = model.fit(x=[X[train]], y=y[train], epochs=200,
                                batch_size=256, validation_split=0.2, verbose=1,
                                shuffle=False, callbacks=[early_stopping])



        list_history.append(history)

        print("\n ### Evaluate (Test data) on Fold : ", current_fold)

        loss, accuracy = model.evaluate(x=[X[test]], y=y[test], batch_size=256, verbose=1)
        print('On Fold %d test loss: %.3f' % (current_fold, loss))
        list_loss.append(loss)  # Loss 

        # Make predictions
        actual_cs = y[test]  # Ground Truth
        predicted_cs = model.predict(x=[X[test]])

        print("Actual CS: ", actual_cs)
        print('On Fold %d test accuracy: %.3f' % (current_fold, accuracy))
        list_acc.append(accuracy)






