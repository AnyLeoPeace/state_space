from models.original_transformer import Transformer
import numpy as np
# from models.transformer import *
from keras import optimizers, losses
import keras.backend as K
from utils import padd_data
import os
import tensorflow as tf

data = np.load('./data/syn_data.npy', allow_pickle = True).item()
X_observations = data['X']
true_states = data['states']

'''Pad data'''
max_len = 50

X_padded = padd_data(X_observations, max_len)
# X_padded[:,:,0] = np.arange(50)
# mask = X_padded[:,:,1:].mean(axis=-1) != 0
states_padded = []

for item in true_states:
    pad_len = max_len - len(item)
    if pad_len > 0:
        states_padded.append(np.array(item + [item[1]] * pad_len))

states_padded = np.array(states_padded)


trans = Transformer(X_padded, X_padded, len_limit=50, d_model=32, d_data=16, d_inner_hid=32, n_head=4)

trans.compile()
model = trans.model

model.fit([X_padded, X_padded], batch_size=128, epochs=100)

p = model.predict([X_padded[:10], X_padded[:10]])

import matplotlib.pyplot as plt

f = plt.figure()
plt.subplot(1,2,1)
plt.plot(p[0][:,0])
plt.subplot(1,2,2)
plt.plot(X_padded[0][:,1])

f.savefig('./fig')
