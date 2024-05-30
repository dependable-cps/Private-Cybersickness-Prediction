from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.backend import sign
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
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

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)

dir='/.../privacy_work/'
dir_model = dir + 'cnn__all/'
dataset='all_dataset'

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)



# experiments
noise_multiplier = 1.2 # for privacy utility tradeoff[2.0,1.8,1.6,1.4,1.2,1.0]
epochs = 200 #150 for Gameplay Dataset
batch_size = 256
learning_rate = 0.001
num_microbatches = 1
l2_norm_clip = 1

#  model construction
# noise_multiplier = 1 and  0.0
# epochs = 1
# batch_size = 12800
# learning_rate = 0.0001
# num_microbatches = 1
# l2_norm_clip = 1

model_loss = []
model_val_loss = []
loss_delta = []

model_acc = []
model_val_acc = []
acc_delta = []

eps_values = []

# MLP training model
def train():

  optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
      l2_norm_clip=l2_norm_clip,
      noise_multiplier=noise_multiplier,
      num_microbatches=num_microbatches,
      learning_rate=learning_rate)

  loss = tf.keras.losses.sparse_categorical_crossentropy(
      from_logits=False, label_smoothing=0.5, axis=-1,
      reduction=tf.losses.Reduction.NONE, name='sparse_categorical_crossentropy'
  )

  # MODEL
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
  # model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])
  model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])

  hist = model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.5, epochs=epochs, shuffle=True)

  score_train, acc_train = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
  score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

  model_loss.append(score_train)
  model_val_loss.append(score_test)
  model_acc.append(acc_train)
  model_val_acc.append(acc_test)
  modelo_json = model.to_json()
  with open(dir_model+"model_mlp_private.json", "w") as json_file:
    json_file.write(modelo_json)
  model.save_weights(dir_model+"model_mlp_private.h5")


  inputP0=dir_model+'input-P0-0_mlp_private'
  os.system("echo \" \" > "+ inputP0)
  i=0
  for w in model.get_weights():
    try:
      if i==0:
        w1 = []
        for filter in range(32):
          weights_cnn1 = []
          for k in range(10):
            weights_cnn1.append(w[k][0][filter])
          w1.append(weights_cnn1)
        w = np.asarray(w1)
      arq=dir_model+"weights"+str(i)+".csv"
      np.savetxt(arq, w.ravel(), delimiter=" ",fmt='%f')
      os.system("cat "+arq+" >> "+ inputP0)
      print(i, np.array(w).shape, np.ravel(w).shape)
      i=i+1
    except:
      print('erro',w.shape)

# plot the accuracy
def testVsTrainAccFigure(model_name):

    x = eps_values

    values = model_acc + model_val_acc
    values.sort()

    y_min = values[0]
    y_max = values[-1]

    y_min = y_min - (y_min * .01)
    y_max = y_max + (y_max * .01)

    # create an index for each tick position
    xi = list(range(len(x)))
    y = model_acc
    y2 = model_val_acc
    plt.ylim(y_min, y_max)
    # plot the index for the x-values
    plt.plot(xi, y, marker='o', linestyle='--', color='r', label='train')
    plt.plot(xi, y2, marker='o', linestyle='--', color='b', label='test')
    plt.xlabel('epsilon')
    plt.ylabel('accuracy')
    plt.xticks(xi, x)
    plot_title = model_name +' Test vs. Train Accuracy'
    plt.title(plot_title)
    plt.legend()
    plt.savefig(dir_model+model_name+"testVsTrainAccuracyFigure.png", dpi=200)
    plt.show()

# plot the difference between testing and training loss/accuracy
def deltaFigure(model_name):

  i = len(model_loss) -1
  while(i >= 0):
    loss_delta.append(abs(model_val_loss[i] - model_loss[i]) * 100)
    i = i - 1


  i = len(model_acc) -1
  while(i >= 0):
    acc_delta.append(abs(model_val_acc[i] - model_acc[i]) * 100)
    i = i - 1

  print(loss_delta)

  x = np.arange(len(eps_values))  # the label locations
  width = 0.35  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, loss_delta, width, label='Train/Test Loss delta')
  rects2 = ax.bar(x + width/2, acc_delta, width, label='Train/Test Acc delta')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  #ax.set_ylabel('Train/Test Delta %')
  ax.set_title(model_name + ' Train/Test Delta %')
  ax.set_xticks(x, eps_values)
  ax.legend()

  # ax.bar_label(rects1, padding=3)
  # ax.bar_label(rects2, padding=3)

  fig.tight_layout()
  fig.savefig(dir_model+model_name+"deltaFigure.png", dpi=200)
  plt.show()

  #computes epsilon based on noise multiplier and batch size and calls main training model

def runTraining(noise):

  noise_multiplier = noise
  eps, rdp = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(n=np.shape(X_train)[0],
                                              batch_size=batch_size,
                                              noise_multiplier=noise_multiplier,
                                              epochs=epochs,
                                              delta=1/np.shape(X_train)[0])
  eps_values.append(round(eps, 3))
  model = train()


# send noise multiplier as argument to runTraining


#epsilon = 1
runTraining(0.82)

#epsilon = 3
runTraining(0.64)

#epsilon = 6
runTraining(0.484)

print(eps_values)

testVsTrainAccFigure("MLP")
