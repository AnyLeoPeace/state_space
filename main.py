### Setting up the experiment 
from data.make_data import generate_trajectory
from models.transformer import *
from keras import optimizers, losses
import keras.backend as K
import numpy as np
from models.SeqModels import padd_data, state_to_array


'''Generate data'''
X_observations, true_states = generate_trajectory(num_states=3, 
                                                  Num_observations=16, 
                                                  Num_samples=10, 
                                                  Max_seq=30, 
                                                  Min_seq=3, 
                                                  alpha=100,
                                                  reverse_mode=False)


'''Pad data'''
max_len = 30

X_padded = padd_data(X_observations, max_len)
states_padded = []

for item in true_states:
    pad_len = max_len - len(item)
    if pad_len > 0:
        states_padded.append(np.array(item + [item[1]] * pad_len))

states_padded = np.array(states_padded)

trans = Transformer(len_limit=30, layers=1, n_head=1)
trans.compile()

trans.model.compile(optimizer='sgd')

trans.model.fit(X_padded, batch_size=1)

trans.model.predict(X_padded[:1])