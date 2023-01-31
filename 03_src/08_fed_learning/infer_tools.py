import pandas as pd
import numpy as np

import json

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append("../")
import utils

################# INFERENCE METHODS ################

def infer_location(model, oracle, baseline_model, parking_testset, test_t, settings, num_best = 5):
    ''' Infering location by comparing the model results to a baseline model.
        Parameters:
         - model: a keras model
         - oracle: oracle dataframe
         - baseline_model: a model to which we shall compare the performance
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
        base_occup = baseline_model.predict(test_d, batch_size=1000, verbose=0).reshape(len(test_t))
        oracle_occup = utils.standardize(oracle[oracle["ids"] == parking].groupby("timestamp")["percentage"].mean(),
                                         settings["mean"], settings["std"])
        loss = np.mean((occup-oracle_occup)**2)
        base_loss = np.mean((occup-base_occup)**2)
        parking_losses.append(loss-base_loss)
        
    best_parkings = np.argsort(parking_losses)[:num_best]+min(plist)
    for i in range(len(best_parkings)):
        if best_parkings[i] > 1148:
            best_parkings[i] += 1
            
    return best_parkings



def infer_time(model, oracle, baseline_model, best_parkings, test_t, settings, window=60):
    ''' Infering moving time.
        Parameters:
         - model: a keras model
         - oracle: oracle dataframe
         - best_parkings: list of the best parking lot ids
         - test_t: normalized testing times
         - settings: a dictionary with min,max,mean,std values
         - window: window for the moving average (measured in minutes)
        Returns: estimated mean moving time in [s]'''
    
    parking_losses_time = []
    plist=[i for i in range(1059,1186+1)]
    parking_test_t = np.tile(test_t, len(best_parkings)).reshape(len(best_parkings)*len(test_t), 1)
    parking_test_id = None
    for b in best_parkings:
        #if not(parking_test_id is None):
            #print(len(parking_test_id))
        #print(len([b]*len(test_t)))
        if parking_test_id is None:
            parking_test_id = utils.one_hot_encoder([b]*len(test_t), plist)
        else:
            parking_test_id = np.vstack([parking_test_id, utils.one_hot_encoder([b]*len(test_t), plist)])
        
    test_d = np.hstack([parking_test_id, parking_test_t])
    occup = model.predict(test_d, batch_size=1000, verbose=0).reshape(len(test_d))
    base_line_occup = baseline_model.predict(test_d, batch_size=1000, verbose=0).reshape(len(test_d))
    
    for t in range(14460, 50400+1, 60):
        #id_encoding = utils.one_hot_encoder(best_parkings, plist)
        #time = np.array([utils.normalize(t, settings["min"], settings["max"])]*len(id_encoding))
        #test_d = np.hstack([id_encoding, time.reshape(len(id_encoding), 1)])
        #occup = model.predict(test_d, batch_size=1000, verbose=0).reshape(len(best_parkings))
        #oracle_occup = []
        base_idx = (t-14460) // 60
        oracle_t = oracle[oracle["timestamp"] == t]
        oracle_occups = []
        pred_occups = []
        base_occups = []
        for i, bp in enumerate(best_parkings):
            oracle_bp = oracle_t[oracle_t["ids"] == bp]
            oracle_occup = utils.standardize(np.mean(oracle_bp["percentage"]), settings["mean"], settings["std"])
            oracle_occups.append(oracle_occup)
            pred_occups.append(occup[base_idx+len(test_t)*i])
            base_occups.append(base_line_occup[base_idx+len(test_t)*i])
            #oracle_occup.append(utils.standardize(oracle_f[oracle_f["timestamp"] == t].groupby("ids")["percentage"].mean(),
            #                                 settings["mean"], settings["std"]))
        #loss = np.sum( ((occup-oracle_occup)**2) < 0.01)
        oracle_occups = np.array(oracle_occups)
        pred_occups = np.array(pred_occups)
        base_occups = np.array(base_occups)
        
        loss_base = np.array(np.mean((oracle_occups-base_occups)**2))
        loss_pred = np.array(np.mean((oracle_occups-pred_occups)**2))
        loss = loss_base-loss_pred
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

def evaluate_performance(model, oracle, baseline_model, parking_testset, test_t, settings,
                         true_parkings, true_time):
    '''
        Evaluates the model.
        Parameters:
             - model: a keras model
             - oracle: oracle dataframe
             - baseline_model: a model to which we shall compare the performance
             - parking_testset: parking lots and timestamps encoded for the neural network
             - test_t: normalized testing times
             - settings: a dictionary with min,max,mean,std values
             - true_parkings: list of true parking lots
             - true_time: true moving time
        Returns:
            accuracy, absolute error
    '''
    
    #print("Calculating best parkings....")
    best_parkings = infer_location(model, oracle, baseline_model, parking_testset, test_t,
                                   settings)
    #print("infering time...")
    time = infer_time(model, oracle, baseline_model, best_parkings, test_t, settings)
    
    return eval_location(best_parkings, true_parkings), eval_time(time, true_time)
    