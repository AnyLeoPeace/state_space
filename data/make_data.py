
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from __future__ import absolute_import, division, print_function

import collections
import hashlib
import numbers
import itertools
import functools
# import sets
import inspect
import pickle

from sklearn.model_selection import *
from sklearn.metrics import *

# import tensorflow as tf
# from tensorflow.contrib.rnn import PhasedLSTMCell, MultiRNNCell, BasicRNNCell
# from tensorflow.python.ops import rnn_cell, rnn
# from tensorflow.python.eager import context
# from tensorflow.python.framework import constant_op, dtypes, ops, tensor_shape, tensor_util   
# from tensorflow.python.layers import base as base_layer
# from tensorflow.python.ops import * 
# from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.training import checkpointable
# from tensorflow.python.util import nest
# from tensorflow.python.util.tf_export import tf_export

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from matplotlib import pyplot as plt
# get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import shutil
import random

def get_attention_weights(seqlength, alpha, reverse_mode):

    init_weights = np.exp(-alpha * np.array(range(seqlength)))

    if reverse_mode:

        weights_ = init_weights / np.sum(init_weights)

    else:

        weights_ = np.flip(init_weights / np.sum(init_weights))
        
    return weights_ 

def generate_trajectory(num_states=3, 
                        Num_observations=10, 
                        Num_samples=1000, 
                        Max_seq=20, 
                        Min_seq=3,
                        alpha=1,
                        reverse_mode=False,
                        P_trans = np.array([[0.9, 0.1, 0.06], 
                                            [0.6, 0.1, 0.3], 
                                            [0.1, 0.8, 0.1]]),
                        P_0=[0.5, 0.3, 0.2],
                        mu_=[-10, 5, 10],
                        var_=[0.5, 1, 1.5]):
    
    X_  = []
    S_  = []
    a_  = []
    
    for k in range(Num_samples):
    
        seq_len = np.random.randint(Min_seq, Max_seq)
        S_new   = []
        a_new   = []
        X_new   = []
    
        for u in range(seq_len):
        
            if u == 0:
            
                S_new.append(np.random.choice(num_states, 1, p=P_0)[0])
            
            else:
            
                weights_    = get_attention_weights(u + 1, alpha, reverse_mode)            
                P_trans_new = np.sum(np.array([P_trans[S_new[m], :] * weights_[m] for m in range(len(S_new))]), axis=0)
                P_trans_new = P_trans_new/np.sum(P_trans_new)
            
                S_new.append(np.random.choice(num_states, 1, p=P_trans_new)[0])
                a_new.append(weights_)
            
            X_new.append((mu_[S_new[-1]] + var_[S_new[-1]] * np.random.normal(0, 1, (1, Num_observations))).reshape(-1,))
            
    
        S_.append(S_new)
        a_.append(a_new)
        X_.append(np.array(X_new))
     
    return X_, S_    



def generate_trajectory_new(num_states=3, 
                        Num_observations=10, 
                        Num_samples=1000, 
                        Max_seq=20, 
                        Min_seq=3,
                        Max_length = 50,
                        alpha = 1,
                        reverse_mode=False,
                        P_trans = np.array([[0.9, 0.1, 0.01], 
                                            [0.3, 0.6, 0.1], 
                                            [0.1, 0.8, 0.1]]),
                        P_0=[0.9, 0.1, 0],
                        mu_=[-10, 5, 10],
                        var_=[1, 1, 1]):
    '''
    alpha: the attention parameter
    beta: the temporal parameter 
    '''

    X_  = []
    S_  = []
    a_  = []
    time_ = []
    mask_ = []
    total = 0

    
    for k in range(Num_samples):
    
        seq_len = np.random.randint(Min_seq, Max_seq-1)
        seq_select = random.sample(list(np.arange(Max_length-1)+1), seq_len-1)
        seq_mask = np.zeros(Max_length)
        seq_mask[seq_select] = True
        seq_mask[0] = True
        time = np.where(seq_mask)[0]
        mask_.append(seq_mask)
        time_.append(time)


        S_new   = []
        a_new   = []
        X_new   = []
    
        for u in range(Max_length):
        
            if u == 0:
            
                S_new.append(np.random.choice(num_states, 1, p=P_0)[0])
            
            else:

                weights_    = get_attention_weights(u + 1, alpha, reverse_mode)            
                P_trans_new = np.sum(np.array([P_trans[S_new[m], :] * weights_[m] for m in range(len(S_new))]), axis=0)
                P_trans_new = P_trans_new/np.sum(P_trans_new)
            
                S_new.append(np.random.choice(num_states, 1, p=P_trans_new)[0])
                a_new.append(weights_)
            
            X_new.append((mu_[S_new[-1]] + var_[S_new[-1]] * np.random.normal(0, 1, (1, Num_observations))).reshape(-1,))
            
    
        S_.append(np.array(S_new)[time])
        a_.append(a_new)
        X_new = np.array(X_new)[time]
        
        total += len(time)
        X_.append(np.array(X_new))
     
    return X_, S_ , time_, total  


def generate_trajectory_final(num_states=3, 
                        Num_observations=10, 
                        Num_samples=1000, 
                        Max_seq=20, 
                        Min_seq=3,
                        Max_length = 50,
                        alpha = 1,
                        proportion = 0.5,
                        reverse_mode=False,
                        P_trans = np.array([[0.9, 0.1, 0.01], 
                                            [0.3, 0.6, 0.1], 
                                            [0.1, 0.8, 0.1]]),
                        P_0=[0.9, 0.1, 0],
                        mu_=[-10, 5, 10],
                        var_=[0.5, 1, 1.5]):
    '''
    Max_length: the time length
    Proportion: the ratio between actual visits and all possible visits
    '''

    X_  = []
    S_  = []
    time_ = []
    total_visit = 0

    # All visit positions
    num_pos = Max_length * Num_samples
    pos_ = random.sample(list(np.arange(num_pos)), int(num_pos * proportion))
    mask_ = np.zeros(num_pos, bool)
    mask_[pos_] = True
    mask_ = mask_.reshape((Num_samples, Max_length))

    
    for k in range(Num_samples):
    
        seq_mask = mask_[k]
        seq_mask[0] = True
        time = np.where(seq_mask)[0]

        while len(time) < Min_seq:
            time = np.unique(list(time) + [np.random.randint(1, Max_length)])
        
        while len(time) >= Max_seq:
            bad = random.sample(list(time),1)
            seq_mask[bad] = False
            seq_mask[0] = True
            time = np.where(seq_mask)[0]

        S_new   = []
        X_new   = []
    
        for u in range(Max_length):
        
            if u == 0:
            
                S_new.append(np.random.choice(num_states, 1, p=P_0)[0])
            
            else:

                weights_    = get_attention_weights(u + 1, alpha, reverse_mode)            
                P_trans_new = np.sum(np.array([P_trans[S_new[m], :] * weights_[m] for m in range(len(S_new))]), axis=0)
                P_trans_new = P_trans_new/np.sum(P_trans_new)
            
                S_new.append(np.random.choice(num_states, 1, p=P_trans_new)[0])
            
            X_new.append((mu_[S_new[-1]] + var_[S_new[-1]] * np.random.normal(0, 1, (1, Num_observations))).reshape(-1,))
            

        S_.append(np.array(S_new)[time])
        X_.append(np.array(X_new)[time])
        time_.append(time)
        total_visit += len(time)

     
    return X_, S_ , time_, total_visit

