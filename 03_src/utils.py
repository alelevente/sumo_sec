#Useful tools for processing the data

import numpy as np



def standardize(x, mean, std):
    return (x-mean)/std

def de_standardize(y, mean, std):
    return y*std+mean

def normalize(x, min_d, max_d):
    return (x-min_d)/(max_d-min_d)

def de_normalize(y, min_d, max_d):
    return y*(max_d-min_d)+min_d

def one_hot_encoder(raw_ids, id_list):
    id_encoding = np.zeros((len(raw_ids), len(id_list)))
    for i,x in enumerate(raw_ids):
        id_encoding[i, id_list.index(x)] = 1.0
    return id_encoding

