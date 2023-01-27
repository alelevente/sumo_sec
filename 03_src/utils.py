#Useful tools for processing/visualizing the data

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json

import sys,os

SUMO_HOME = os.environ["SUMO_HOME"] #locating the simulator
sys.path.append(SUMO_HOME+"/tools")
import sumolib
from sumolib.visualization import helpers


############################################
####### DATA CONVERSION FUNCTIONS ########

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


###########################################
######### VISUALIZATION FUNCTIONS ########

class Option:
    #default options required by sumolib.visualization
    defaultWidth = 1.5
    defaultColor = (0.0, 0.0, 0.0, 0.0)

def plot_network_probs(net, probabilities, index_to_edge_map, cmap="YlGn",
                       title="", special_edges=None,
                       special_color=(1.0, 0.0, 0.0, 1.0),
                       fig=None, ax=None):

    '''
        Plots a road network with edges colored according to a probabilistic distribution.
        Parameters:
            net: a sumolib road network
            probabilities: a dictionary that maps edge indices to probabilities
                If an edge is not in this map, it will get a default (light gray) color.
            index_to_edge_map: a dictionary that maps edge indices to SUMO edge IDs
            cmap: the colormap to be used on the plot
            title: title of the produced plot
            special_edges: edges to be plotted with special color, given in a similar structure
                as probabilities parameter
            special_color: color of the special edges (RGBA)
            fig: if None then a new map is created; if it is given, then only special edges are overplot to the original fig
            ax: see fig

        Returns:
            a figure and axis object
    '''
        
    scalar_map = None
    colors = {}
    options = Option()
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(22, 20))
        if probabilities is None:
            for e in index_to_edge_map.values():
                colors[e] = (0.125, 0.125, 0.125, .25)
        else:
            c_norm = matplotlib.colors.LogNorm(vmin=min(probabilities)*0.85, vmax=max(probabilities)*1.15)
            scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
            for i,p in enumerate(probabilities):
                if (p == 0.0) and (i in index_to_edge_map):
                    colors[index_to_edge_map[i]] = (0.125, 0.125, .125, .125)
                elif i in index_to_edge_map:
                    colors[index_to_edge_map[i]] = scalar_map.to_rgba(min(1,max(0,p)))
                    
    if not(special_edges is None):
        for ind in special_edges:
            colors[index_to_edge_map[ind]] = special_color
    
    helpers.plotNet(net, colors, [], options)
    plt.title(title)
    plt.xlabel("position [m]")
    plt.ylabel("position [m]")
    ax.set_facecolor("lightgray")
    if not(scalar_map is None):
        plt.colorbar(scalar_map)

    return fig, ax