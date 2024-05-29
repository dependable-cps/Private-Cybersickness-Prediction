# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""


import math
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""Generate data from the data source Simulation 2021 dataset"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, base_path, 
                 batch_size=256, time_step_time_series=90, classification=False, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.classification = classification
        self.base_path = base_path
        self.indexes = dataset.index
        self.classes = None
        self.tmp = []
        self.time_step_for_time_series = time_step_time_series
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.dataset) // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_index = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in batch_index]
        # Process Eye-tracking data
        eye = self.dataset.loc[list_IDs_temp, ['eye']].to_numpy()
        eye = '/eye' + eye
        eye = self.prepare_time_series_data(data_path=eye)
        eye = eye.reshape((self.batch_size, -1, 60, 9))

        # Process Head-tracking data
        head = self.dataset.loc[list_IDs_temp, ['head']].to_numpy()
        head = '/head' + head
        head = self.prepare_time_series_data(data_path=head)
        head = head.reshape((self.batch_size, -1, 60, 4))
        
        #Labels generation for cybersickness
        cs_class = self.dataset.loc[list_IDs_temp, ['cs_class']].to_numpy()
        self.tmp.extend(cs_class)
        cs_class = tf.keras.utils.to_categorical(cs_class, num_classes=4)
        return  cs_class

    def on_epoch_end(self):
        """Updates the indexes after each epoch"""
        self.classes = self.tmp
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def prepare_time_series_data(self, data_path=None):
        X = []
        for path in data_path:
            current_path = self.base_path + path
            # print(current_path)
            data = pd.read_csv(current_path[0])
            data = data.drop(['#Frame', 'cs'], axis=1)  # Drop the frame number from the dataset
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            for i in range(data.shape[0]):
                end_ix = i + self.time_step_for_time_series
                if end_ix >= data.shape[0]:
                    break
                seq_X = data[i:end_ix]       
                X.append(seq_X)
        sample_per_batch = random.sample(X, self.batch_size)
        return np.array(sample_per_batch)
