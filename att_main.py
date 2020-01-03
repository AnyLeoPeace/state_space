### Setting up the experiment 
from data.make_data import generate_trajectory
from models.SeqModels import *
from keras import optimizers, losses
import keras.backend as K
import numpy as np
from keras_radam import RAdam
from scipy.stats import multivariate_normal
from keras.utils import np_utils
import os
from utils import *
# X_observations, true_states = generate_trajectory(num_states=3, 
                                                  # Num_observations=10, 
                                                  # Num_samples=100, 
                                                  # Max_seq=30, 
                                                  # Min_seq=3, 
                                                  # alpha=100,
                                                  # reverse_mode=False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


max_seq = 15
min_seq = 5
max_length = 30
alpha = 100

data = np.load('./data/syn_data_' + str(max_seq) + '_'  + str(min_seq) + '_' + str(max_length) + '_' + str(alpha) + '.npy', allow_pickle = True).item()
X_observations = data['X']
true_states = data['states']
X_time = data['time']

### Training attentive state-space models
tf.reset_default_graph()
model = attentive_state_space_model(num_states=3,
                              maximum_seq_length=max_seq, 
                              input_dim=10, 
                              rnn_type='LSTM',
                              latent=True,
                              generative=True,
                              num_iterations=50, 
                              num_epochs=100, 
                              batch_size=100, 
                              learning_rate=5*1e-4, 
                              num_rnn_hidden=100, 
                              num_rnn_layers=1,
                              dropout_keep_prob=None,
                              num_out_hidden=100, 
                              num_out_layers=1)


model.fit(X_observations)


data_eval = np.load('./data/syn_data_eval' + str(max_seq) + '_'  + str(min_seq) + '_' + str(max_length) + '_' + str(alpha) + '.npy', allow_pickle = True).item()
X_observations_eval = data_eval['X']
true_states_eval = data_eval['states']
X_time_eval = data_eval['time']

state_inference, expected_observations, attention = model.predict(X_observations_eval) # Here states has delay

corr = 0
l = 0
all_true = []
all_seq = []
seqs = []
for index, item in enumerate(state_inference):
    seq = item.argmax(-1)
    seqs.append(seq)
    all_true = all_true + list(true_states_eval[index][1:]) # delay
    all_seq = all_seq + list(seq[:-1]) # no last one
    l += len(seq)

all_true = np.array(all_true)
all_seq = np.array(all_seq)

print('Acc :',(label_exchange(1,0,all_true) == all_seq).mean())
print('F1 Score :',f1_score(all_true, all_seq, average='macro'))


X_ = padd_data(X_observations_eval, model.maximum_seq_length)[:,1:].reshape(-1, 10) # X has a delay
mask = X_.mean(axis=-1) != 0
X_ = X_[mask]
pred = np_utils.to_categorical(all_seq, num_classes=model.num_states)

lks_  = np.array([multivariate_normal.logpdf(X_, model.state_means[k], model.state_covars[k]).reshape((-1,1)) * pred[:,k].reshape((-1,1)) for k in range(model.num_states)])
lks = np.mean(lks_)
print('lks',lks)