### Setting up the experiment 
# from data.make_data import generate_trajectory
# from keras import optimizers, losses
# import keras.backend as K
import numpy as np
from utils import padd_data
import os
# import tensorflow as tf

'''Generate data'''
# X_observations, true_states = generate_trajectory(num_states=3, 
#                                                   Num_observations=16, 
#                                                   Num_samples=5000, 
#                                                   Max_seq=50, 
#                                                   Min_seq=30, 
#                                                   alpha=100,
#                                                   reverse_mode=False)

# np.save('./data/syn_data.npy',{'X':X_observations, 'states':true_states})

data = np.load('./data/syn_data.npy', allow_pickle = True).item()
X_observations = data['X']
true_states = data['states']

'''Pad data'''
max_len = 50

X_padded = padd_data(X_observations, max_len)
X_padded[:,:,0] = np.arange(50)
mask = X_padded[:,:,1:].mean(axis=-1) != 0
states_padded = []

for item in true_states:
    pad_len = max_len - len(item)
    if pad_len > 0:
        states_padded.append(np.array(item + [item[1]] * pad_len))

states_padded = np.array(states_padded)



train_X = []
train_states = []
train_masks = []

for index, item in enumerate(X_padded):
    for i in range(len(item)):
        if mask[i] is True:
            temp_mask = np.zeros(len(item),bool)
            temp_mask[:i+1] = 1
            seq = np.copy(item) * temp_mask
            states = np.copy(states_padded[index]) * temp_mask
            train_X.append(seq)
            train_states.append(states)
            train_masks.append(temp_mask)
        else:
            break






