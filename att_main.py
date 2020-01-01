### Setting up the experiment 
from data.make_data import generate_trajectory
from models.SeqModels import attentive_state_space_model as attentive_state_space
from keras import optimizers, losses
import keras.backend as K
import numpy as np


X_observations, true_states = generate_trajectory(num_states=3, 
                                                  Num_observations=10, 
                                                  Num_samples=100, 
                                                  Max_seq=30, 
                                                  Min_seq=3, 
                                                  alpha=100,
                                                  reverse_mode=False)

### Visualizing the true data trajectories
# from matplotlib import pyplot as plt

# trajectory_index = 1

# fig, ax1 = plt.subplots()

# t = list(range(len(true_states[trajectory_index])))

# color = 'tab:red'
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Observations', color=color)
# ax1.plot(t, X_observations[trajectory_index], color=color, linewidth=5, alpha=0.2)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  

# color = 'tab:blue'
# ax2.set_ylabel('Hidden states', color=color)  
# ax2.step(t, true_states[trajectory_index], color=color, linewidth=5)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_yticks([0, 1, 2])

# fig.tight_layout()  
# plt.show()


### Training attentive state-space models
network = attentive_state_space(num_states=3,
                              maximum_seq_length=30, 
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

network.build_transformer_model(transformer_depth = 4, transformer_dropout = 0.1, num_heads = 2, confidence_penalty_weight = 0.1)
transformer_model = network.transformer_model

# generate states by GM
X_, states_ = network.prepare_data(X_observations)
states = states_.argmax(-1)
states = np.expand_dims(states, axis=-1)


def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))

learning_rate = 0.001

optimizer = optimizers.Adam(
                lr=learning_rate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)

transformer_model.compile(
                optimizer,
                loss=losses.sparse_categorical_crossentropy,
                metrics=[perplexity])

transformer_model.fit(X_, states)
