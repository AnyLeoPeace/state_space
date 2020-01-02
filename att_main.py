### Setting up the experiment 
from data.make_data import generate_trajectory
from models.SeqModels import *
from keras import optimizers, losses
import keras.backend as K
import numpy as np
from keras_radam import RAdam



# X_observations, true_states = generate_trajectory(num_states=3, 
#                                                   Num_observations=10, 
#                                                   Num_samples=5000, 
#                                                   Max_seq=50, 
#                                                   Min_seq=30, 
#                                                   alpha=100,
#                                                   time=False,
#                                                   reverse_mode=False)
                                    
# np.save('./data/syn_data_no_time.npy',{'X':X_observations, 'states':true_states})


data = np.load('./data/syn_data_no_time.npy', allow_pickle = True).item()
X_observations = data['X']
true_states = data['states']

### Training attentive state-space models
network = attentive_state_space_model(num_states=3,
                              maximum_seq_length=50, 
                              input_dim=10, 
                              rnn_type='LSTM',
                              inference_network = 'transformer',
                              latent=True,
                              generative=True,
                              num_iterations=50, 
                              num_epochs=3, 
                              batch_size=100, 
                              learning_rate=5*1e-4, 
                              num_rnn_hidden=100, 
                              num_rnn_layers=1,
                              dropout_keep_prob=None,
                              num_out_hidden=100, 
                              num_out_layers=1)


# generate states by GM
X_, states_ = network.prepare_data(X_observations)
states = states_.argmax(-1)
# states = np.expand_dims(states, axis=-1)
X_time = np.repeat(np.expand_dims(np.arange(50), axis=0),5000, axis=0)

network.build_transformer_model(transformer_depth = 4, transformer_dropout = 0.1, num_heads = 2, confidence_penalty_weight = 0.1)
transformer_model = network.transformer_model



# Training
learning_rate = 0.001
optimizer = RAdam(lr=learning_rate)

transformer_model.compile(
                optimizer,
                metrics=['accuracy'])

transformer_model.fit([X_, X_time, states], epochs=10)
