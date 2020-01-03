### Setting up the experiment 
from data.make_data import generate_trajectory
from models.origin_seq import attentive_state_space_model
from keras import optimizers, losses
import keras.backend as K
import numpy as np
from sklearn.metrics import f1_score

X_observations, true_states = generate_trajectory(num_states=3, 
                                                  Num_observations=10, 
                                                  Num_samples=100, 
                                                  Max_seq=30, 
                                                  Min_seq=3, 
                                                  alpha=100,
                                                  reverse_mode=False)

model = attentive_state_space_model(num_states=3,
                              maximum_seq_length=30, 
                              input_dim=10, 
                              rnn_type='LSTM',
                              latent=True,
                              generative=True,
                              num_iterations=100, 
                              num_epochs=500, 
                              batch_size=100, 
                              learning_rate=5*1e-4, 
                              num_rnn_hidden=100, 
                              num_rnn_layers=1,
                              dropout_keep_prob=None,
                              num_out_hidden=100, 
                              num_out_layers=1)


model.fit(X_observations)
state_inference, expected_observations, attention = model.predict(X_observations)

corr = 0
l = 0
all_true = []
all_seq = []
seqs = []
for index, item in enumerate(state_inference):
    seq = item.argmax(-1)
    seqs.append(seq)
    corr += np.array(seq == true_states[index]).sum()
    all_true = all_true + list(true_states[index])
    all_seq = all_seq + list(seq)
    l += len(seq)

print('Acc :', corr/l)
print('F1 Score :',f1_score(all_true, all_seq, average='micro'))
