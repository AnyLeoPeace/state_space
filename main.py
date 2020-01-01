### Setting up the experiment 
# from data.make_data import generate_trajectory
from models.transformer import *
from keras import optimizers, losses
import keras.backend as K
import numpy as np
from utils import padd_data
import os
import tensorflow as tf

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

'''GPU'''
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8


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


'''Get model'''
trans = Transformer(len_limit=50, time_limit=50, layers=2, n_head=2, d_data=16, d_model=32)
lr_scheduler = LRSchedulerPerStep(trans.d_model, 4000) 
# model_saver = ModelCheckpoint(mfile, monitor='ppl', save_best_only=True, save_weights_only=True)

trans.compile()

'''Init'''
trans.init_model_stage_one(X_padded, batch_size = 128)

trans.recon_model.compile(SGD(1e-3))
trans.recon_model.fit(X_padded, batch_size = 128, epochs=50)



# pred = trans.recon_model.predict(X_padded[:1])

trans.init_model_stage_two(X_padded)

'''Train'''
trans.justify_model(lr=1e-3, loss_weights=[1,0,0])
trans.model.fit(X_padded, batch_size=128, epochs = 50)

trans.model.compile(optimizer=Adam(1e-3), metrics = ['acc'])
results = trans.model.predict(X_padded)
results = np.argmax(results,-1) # All zero???


model = Model(trans.input_xt, trans.enc_mask)
model.predict(X_padded[:10])[0]