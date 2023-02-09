import pandas as pd
import numpy as np

import json

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append("../")
import utils

TEST_T_LEN = 600

################# INFERENCE METHODS ################

def infer_location(occupation, base_occupation, oracle, settings, num_best = 5):
    ''' Infering location by comparing the model results to a baseline model.
        Parameters:
         - occupation: predicted occupation values
         - base_occupation: occupation values provided by the baseline (federated) model
         - oracle: oracle dataframe
         - baseline_model: a model to which we shall compare the performance
         - settings: a dictionary with min,max,mean,std values + parking_ids
         - num_best: how many parking lots to list in the output
        Returns:
         - best infered parking lots'''
       
    parking_losses = []
    for i,parking in enumerate(settings["parkings"]):
        occup = occupation[i*TEST_T_LEN:(i+1)*TEST_T_LEN]
        base_occup = base_occupation[i*TEST_T_LEN:(i+1)*TEST_T_LEN]
        oracle_occup = utils.standardize(oracle[oracle["ids"] == parking].groupby("timestamp")["percentage"].mean(),
                                         settings["mean"], settings["std"])
        loss = np.mean((occup-oracle_occup)**2)
        base_loss = np.mean((occup-base_occup)**2)
        parking_losses.append(loss-base_loss)
        
    best_parkings = np.argsort(parking_losses)[:num_best]+min(settings["parkings"])
    for i in range(len(best_parkings)):
        if best_parkings[i] > 1148:
            best_parkings[i] += 1
            
    return best_parkings



def infer_time(occupation, base_occupation, oracle, best_parkings, settings, window=60):
    ''' Infering moving time.
        Parameters:
         - occupation: predicted occupation values
         - base_occupation: occupation values provided by the baseline (federated) model
         - oracle: oracle dataframe
         - best_parkings: list of the best parking lot ids
         - settings: a dictionary with min,max,mean,std values
         - window: window for the moving average (measured in minutes)
        Returns: estimated mean moving time in [s]'''
  
    parking_losses_time = []
    
    for t in range(14460, 50400+1, 60):
        base_idx = (t-14460) // 60
        oracle_t = oracle[oracle["timestamp"] == t]
        oracle_occups = []
        pred_occups = []
        base_occups = []
        for i, bp in enumerate(best_parkings):
            if bp > 1148: bp -= 1
            oracle_bp = oracle_t[oracle_t["ids"] == bp]
            oracle_occup = utils.standardize(np.mean(oracle_bp["percentage"]), settings["mean"], settings["std"])
            oracle_occups.append(oracle_occup)
            occup = occupation[(bp-min(settings["parkings"]))*TEST_T_LEN : (bp-min(settings["parkings"])+1)*TEST_T_LEN]
            base_line_occup = base_occupation[(bp-min(settings["parkings"]))*TEST_T_LEN : (bp-min(settings["parkings"])+1)*TEST_T_LEN]
            if len(occup) == 0:
                print((bp-min(settings["parkings"]))*TEST_T_LEN, (bp-min(settings["parkings"])+1)*TEST_T_LEN)
            pred_occups.append(occup[base_idx])
            base_occups.append(base_line_occup[base_idx])
            #oracle_occup.append(utils.standardize(oracle_f[oracle_f["timestamp"] == t].groupby("ids")["percentage"].mean(),
            #                                 settings["mean"], settings["std"]))
        #loss = np.sum( ((occup-oracle_occup)**2) < 0.01)
        oracle_occups = np.array(oracle_occups)
        pred_occups = np.array(pred_occups)
        base_occups = np.array(base_occups)
        
        loss_base = np.array(np.mean((oracle_occups-base_occups)**2))
        loss_pred = np.array(np.mean((oracle_occups-pred_occups)**2))
        #loss = loss_base-loss_pred
        loss = loss_pred-loss_base
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
    
    occup = model.predict(parking_testset, verbose=0).reshape(len(parking_testset))
    base_line_occup = baseline_model.predict(parking_testset, batch_size=1000, verbose=0).reshape(len(parking_testset))
    
    #print("Calculating best parkings....")
    best_parkings = infer_location(occup, base_line_occup, oracle, settings)
    #print("infering time...")
    time = infer_time(occup, base_line_occup, oracle, best_parkings, settings)
    
    return eval_location(best_parkings, true_parkings), eval_time(time, true_time)
    