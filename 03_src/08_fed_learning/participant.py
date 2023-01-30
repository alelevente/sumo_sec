import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

import sys
sys.path.append("../")
from neural_network import NeuralNetwork

MEAS_ROOT = "../../02_data/vehicle_data/"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


class FLParticipant:
    def __init__(self):
        self.nn = NeuralNetwork()
        
    def train(self, weights, veh_num, day, epochs=1):
        #load data until the given day:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
        train_data = None
        for d in range(day+1):
            #print("loading: "+MEAS_ROOT+"_%d.0_%d.csv"%(veh_num, d))
            try:
                td = pd.read_csv(MEAS_ROOT+"_%s.0_%d.csv"%(veh_num, d), header=None)
                if train_data is None:
                    train_data = td
                else:
                    train_data = pd.concat([train_data, td])
            except:
                pass
        
        train_data = np.array(train_data.sample(frac=1.0))
        #print(train_data.shape)
        
        self.nn.model.set_weights(weights)
        self.nn.model.fit(train_data[:,:-2], train_data[:,-1], epochs=epochs,
                          verbose = 0, callbacks=[callback])
        return self.nn.model.get_weights(), len(train_data)
        

