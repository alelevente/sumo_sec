import pandas as pd
import numpy as np

import json

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append("../")
import utils

################# INFERENCE METHODS ################

def infer_location(model, oracle, parking_testset, test_t, settings, num_best = 5):
    ''' Infering location.
        Parameters:
         - model: a keras model
         - oracle: oracle dataframe
         - parking_testset: parking lots and timestamps encoded for the neural network
         - test_t: normalized testing times
         - settings: a dictionary with min,max,mean,std values
         - num_best: how many parking lots to list in the output
        Returns:
         - best infered parking lots'''
    
    plist=[i for i in range(1059,1186+1)]
    plist.remove(1148)
    
    parking_losses = []
    for i,parking in enumerate(plist):
        test_d = parking_testset[i*len(test_t):(i+1)*len(test_t)]
        occup = model.predict(test_d, batch_size=1000, verbose=0).reshape(len(test_t))
        oracle_occup = utils.standardize(oracle[oracle["ids"] == parking].groupby("timestamp")["percentage"].mean(),
                                         settings["mean"], settings["std"])
        loss = np.mean((occup-oracle_occup)**2)
        parking_losses.append(loss)
        
    best_parkings = np.argsort(parking_losses)[:num_best]+min(plist)
    for i in range(len(best_parkings)):
        if best_parkings[i] > 1148:
            best_parkings[i] += 1
            
    return best_parkings



def infer_time(model, oracle, best_parkings, settings, window=60):
    ''' Infering moving time.
        Parameters:
         - model: a keras model
         - oracle: oracle dataframe
         - best_parkings: list of the best parking lot ids
         - settings: a dictionary with min,max,mean,std values
         - window: window for the moving average (measured in minutes)
        Returns: estimated mean moving time in [s]'''
    
    parking_losses_time = []
    plist=[i for i in range(1059,1186+1)]

    for t in range(14460, 50400+1, 60):
        id_encoding = utils.one_hot_encoder(best_parkings, plist)
        time = np.array([utils.normalize(t, settings["min"], settings["max"])]*len(id_encoding))
        test_d = np.hstack([id_encoding, time.reshape(len(id_encoding), 1)])
        occup = model.predict(test_d, batch_size=1000, verbose=0).reshape(len(best_parkings))
        oracle_occup = []
        for bp in best_parkings:
            oracle_f = oracle[oracle["ids"] == bp]
            oracle_occup.append(utils.standardize(oracle_f[oracle_f["timestamp"] == t].groupby("ids")["percentage"].mean(),
                                             settings["mean"], settings["std"]))
        #loss = np.sum( ((occup-oracle_occup)**2) < 0.01)
        oracle_occup = np.array(oracle_occup)
        loss = np.array(np.mean((occup-oracle_occup)**2))
        #loss = np.std(occup)
        if loss.shape == ():
            parking_losses_time.append(loss)
        else:
            parking_losses_time.append(loss[0])
        
    #smoothing:
    parking_losses_cv = np.convolve(parking_losses_time, np.ones(window))/window
    parking_losses_cv = parking_losses_cv[window:]
    parking_losses_cv = parking_losses_cv[:-window]

    predt = (np.argmin(parking_losses_cv)+4*60+window)*60
    return predt
    
    
    
############## EVALUATING RESULTS ###################x

def eval_location(best_parkings, true_parkings):
    '''
        Evaluating location inference
        Parameters:
            - best_parkings: list of infered parking lots
            - true_parkings: true parking lots
        Returns:
            - accuracy: found/total
    '''
    tp = 0
    for p in best_parkings:
        if p in true_parkings:
            tp += 1
    return tp/min(5,len(true_parkings))

def eval_time(infered_time, true_time):
    '''
        Evaluating time inference
        Parameters:
            - infered_time: infered time [s]
            - true_time: true moving time [s]
        Returns:
            - absolute error
    '''
    return np.abs(infered_time-true_time)

def evaluate_performance(model, oracle, parking_testset, test_t, settings,
                         true_parkings, true_time):
    '''
        Evaluates the model.
        Parameters:
             - model: a keras model
             - oracle: oracle dataframe
             - parking_testset: parking lots and timestamps encoded for the neural network
             - test_t: normalized testing times
             - settings: a dictionary with min,max,mean,std values
             - true_parkings: list of true parking lots
             - true_time: true moving time
        Returns:
            accuracy, absolute error
    '''
    
    best_parkings = infer_location(model, oracle, parking_testset, test_t,
                                   settings)
    time = infer_time(model, oracle, best_parkings, settings)
    
    return eval_location(best_parkings, true_parkings), eval_time(time, true_time)
    