from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.backend import sign
from tensorflow.keras import optimizers
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
from art.attacks.inference.membership_inference import ShadowModels
from art.utils import to_categorical
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification.pytorch import PyTorchClassifier
import onnx
from onnx2pytorch import ConvertModel
import tf2onnx

# we should specify shape of the input tensor
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


target_train_size_cs = len(X_train) // 2
x_target_train_cs = X_train[:target_train_size_cs]
y_target_train_cs = y_train[:target_train_size_cs]
x_target_test_cs = X_test[target_train_size_cs:]
y_target_test_cs = y_test[target_train_size_cs:]

# LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=128, output_dim=128, input_length=20))
model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(1,20)))
model.add(Dropout(0.15))
model.add(tf.keras.layers.LSTM(units=128), return_sequences = True)
model.add(Dropout(0.15))
model.add(tf.keras.layers.LSTM(units=128), return_sequences = True)
model.add(Dropout(0.15))
model.add(tf.keras.layers.LSTM(units=128), return_sequences = True)
model.add(Dropout(0.15))
model.add(tf.keras.layers.LSTM(units=128), return_sequences = True)
model.add(Dropout(0.15))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.summary()

adam_modified = optimizers.Adam(learning_rate=0.001, beta_1=0.7, beta_2=0.9, amsgrad=False)

model.compile(loss="categorical_crossentropy", optimizer=adam_modified, metrics=["accuracy"])
model.fit(x_target_train_cs, y_target_train_cs, epochs=200)
model.save('../LSTM_model.h5')

#Load the trained cybersickness  model in Keras

loaded_model = tf.keras.models.load_model("LSTM_model.h5")

# Convert model to PyTorch
 
# Convert the model to ONNX format
onnx_CS_model, _ = tf2onnx.convert.from_keras(model)

cybersickness_model_pytorch = ConvertModel(onnx_CS_model)
LSTM_art_model = PyTorchClassifier(model=cybersickness_model_pytorch, input_shape=(42,), nb_classes=4)

train_pred = np.array([np.argmax(arr) for arr in LSTM_art_model.predict(X_train.astype(np.float32))])
print('Base model Train accuracy: ', np.sum(train_pred == y_train) / len(y_train))

test_pred = np.array([np.argmax(arr) for arr in LSTM_art_model.predict(X_test.astype(np.float32))])
print('Base model Test accuracy: ', np.sum(test_pred == y_test) / len(y_test))



##The black-box attack basically trains an additional classifier (called the attack model) to predict the membership status of a sample. 
#It can use as input to the learning process probabilities/logits or losses, depending on the type of model and provided configuration


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

attack_train_ratio = 0.5
attack_train_size = int(len(X_train) * attack_train_ratio)
attack_test_size = int(len(y_test) * attack_train_ratio)

bb_attack = MembershipInferenceBlackBox(LSTM_art_model)

# train attack model
bb_attack.fit(X_train[:attack_train_size], y_train[:attack_train_size],
              X_test[:attack_test_size], y_test[:attack_test_size])



mlp_attack_bb = MembershipInferenceBlackBox(LSTM_art_model, attack_model_type='rf')

# train attack model
mlp_attack_bb.fit(X_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
              X_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

# infer
mlp_inferred_train_bb = mlp_attack_bb.infer(X_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
mlp_inferred_test_bb = mlp_attack_bb.infer(X_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

# check accuracy
mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))

print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
print(f"Attack Accuracy {mlp_acc_bb:.4f}")

print(calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)),
                            np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb))))))



#Membership inference attack with shadow models
shadow_models = ShadowModels(model, num_shadow_models=20)

shadow_dataset = shadow_models.generate_shadow_dataset(X_test, to_categorical(y_test, 4))
(member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

# Shadow models' accuracy
print([sm.model.score(x_target_test_cs, y_target_test_cs) for sm in shadow_models.get_shadow_models()])


attack = MembershipInferenceBlackBox(model)
attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)


member_infer = attack.infer(x_target_train_cs, y_target_train_cs)
nonmember_infer = attack.infer(x_target_test_cs, y_target_test_cs)
member_acc = np.sum(member_infer) / len(x_target_train_cs)
nonmember_acc = 1 - np.sum(nonmember_infer) / len(x_target_test_cs)
acc = (member_acc * len(x_target_train_cs) + nonmember_acc * len(x_target_test_cs)) / (len(x_target_train_cs) + len(x_target_test_cs))
print('Attack Member Acc:', member_acc)
print('Attack Non-Member Acc:', nonmember_acc)
print('Attack Accuracy:', acc)

print(calc_precision_recall(np.concatenate((member_infer, nonmember_infer)),
                            np.concatenate((np.ones(len(member_infer)), np.zeros(len(nonmember_infer))))))