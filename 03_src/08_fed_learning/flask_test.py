from flask import Flask, request

import sys
sys.path.append("..")

import neural_network
import json

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)



app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/weights")
def generate_weights():
    nn = neural_network.NeuralNetwork()
    weights = {"weights": nn.model.to_json()}
    weights = json.dumps(weights)
    return(weights)

@app.route("/get_params", methods=["POST"])
def test_parameters():
    print(request.get_data())
    parameters = json.loads(request.get_data(), strict=False)
    print(parameters)
    return json.dumps({"success": True})
    

@app.route("/train")
def train_model():
    nn = neural_network.NeuralNetwork()
    data = pd.read_csv("../../02_data/vehicle_data/_12110.0_0.csv", header=None)
    train_data = np.array(data)
    nn.model.fit(train_data[:,:-2], train_data[:,-1], epochs=500, verbose = 0)
    return(nn.model.to_json())