import tensorflow as tf
from tensorflow import keras

class NeuralNetwork:
    def __init__(self):
        n_input = keras.Input(shape=(129,))
        self.model = keras.Sequential([n_input,
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(1, activation="linear")])
        
        #self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer="rmsprop", loss="mse")
            
        