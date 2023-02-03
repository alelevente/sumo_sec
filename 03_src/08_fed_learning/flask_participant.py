from flask import Flask, request

import sys
sys.path.append("..")

import neural_network
import json

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

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
        self.nn = neural_network.NeuralNetwork()
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
        
    def train(self, model, veh_num, day, epochs=1):
        #load data until the given day:
        
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
        num_samples = len(train_data)
        #print(train_data.shape)
        
        self.nn.model = keras.models.model_from_json(model)
        self.nn.model.compile(optimizer="rmsprop", loss="mse")
        self.nn.model.fit(train_data[:,:-2], train_data[:,-1], epochs=epochs,
                          verbose = 0, callbacks=[self.callback])
        del train_data
        return self.nn.model.to_json(), num_samples
    
    def eval_model(self, test_data):
        return self.nn.model.predict(test_data, batch_size=1000, verbose=0).reshape(len(test_data))

app = Flask(__name__)
participant = FLParticipant()


@app.route("/train", methods=["POST"])
def train_and_predict():
    parameters = json.loads(request.get_data())
    model_json = parameters["model"]
    veh_num = parameters["vehicle"]
    day = parameters["day"]
    test_data = np.array(parameters["test_data"])
    
    model_json, num_samples = participant.train(model_json, veh_num, day, epochs=50)
    results = participant.eval_model(test_data)
    results = results.tolist()
    
    answer = {
        "model": model_json,
        "num_samples": num_samples,
        "eval_results": results
    }

    return json.dumps(answer)