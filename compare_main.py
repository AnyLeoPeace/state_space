### Setting up the experiment 

from data.make_data import *
import numpy as np
from utils import *
import os
from models.my_model import *
from keras_radam import RAdam
from sklearn.metrics import f1_score
from scipy.stats import multivariate_normal
from keras.utils import np_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

'''Generate data'''
def generate_data(max_seq = 25, min_seq = 5, max_length = 30, alpha = 100, proportion = 0.5):

    X_observations, true_states, X_time, total_visit = generate_trajectory_new(num_states=3, 
                                                        Num_observations=10, 
                                                        Num_samples=1000, 
                                                        Max_seq=max_seq, 
                                                        Min_seq=min_seq, 
                                                        Max_length=max_length,
                                                        alpha=alpha,
                                                        proportion = proportion,
                                                        P_trans = np.array([[0.85, 0.1, 0.05], 
                                                                            [0.1, 0.7, 0.2], 
                                                                            [0, 0.2, 0.8]]),
                                                        reverse_mode=False)
    
    return X_observations, true_states, X_time, total_visit


'''Prepare data'''
def get_train_data(X_padded, states_padded, X_time, len_limit = 25, time_limit = 30):
    train_X = []
    train_states = []
    train_masks = []
    train_time = []
    mask = X_padded[:,:].mean(axis=-1) != 0
    lens = mask.sum(axis=-1)

    for index, item in enumerate(X_padded):
        for i in range(len(item)-1):

            if mask[index][i] == True:
                temp_mask = np.zeros((len(item),1),bool)
                temp_mask[:i+1] = 1

                # generate sample
                seq = np.copy(item) * temp_mask
                states = states_padded[index]
                times = X_time[index][:i+2]
                times = list(times) + [time_limit-1] * (len_limit-len(times))

                # apeend
                train_X.append(seq)
                train_states.append(states)
                train_time.append(times)

                # generate predict mask
                temp_mask[:i+1] = 0
                # temp_mask[i+1:lens[index]] = 1 # Here control how many future state to predict
                temp_mask[i+1] = 1 # Here control how many future state to predict
                train_masks.append(temp_mask[:,0])

            else:
                break
    
    train_X = np.array(train_X)
    train_states = np.array(train_states)
    train_masks = np.array(train_masks)
    train_time = np.array(train_time)

    return train_X, train_time, train_states, train_masks


def train_evaluate_my_model(train_data, eval_data, save = None):
    '''
    Input data directly comes from generate_data_final
    '''

    # Get data
    X_observations, true_states, X_time, total_visit = train_data
    X_observations_eval, true_states_eval, X_time_eval, total_visit_eval = eval_data

    # Get Model
    trans = TranModel(len_limit=25, time_limit=30, d_data=10, d_model=16, d_inner_hid=16, n_head=2, layers=2, num_states=3, dropout=0.3)
    trans.build_transformer_model()
    trans.model.compile(optimizer=RAdam(1e-3))

    # Prepare data
    X_padded, states_padded = trans.prepare_data(X_observations) # Note here states has no delay
    X_padded_eval,states_padded_eval = trans.prepare_data(X_observations_eval, training = False)

    train_X, train_time, train_states, train_masks = get_train_data(X_padded, states_padded, X_time, trans.len_limit, trans.time_limit)
    eval_X, eval_time, eval_states, eval_masks = get_train_data(X_padded_eval, states_padded_eval, X_time_eval, trans.len_limit, trans.time_limit)

    lens = (X_padded[:,:].mean(axis=-1) != 0).sum(axis=-1)
    eval_lens = (X_padded_eval[:,:].mean(axis=-1) != 0).sum(axis=-1)

    # Train
    his = trans.model.fit([train_X, train_time, train_states, train_masks], batch_size=100, epochs=100, verbose=0)

    if save is not None:
        trans.model.save_weights(save + 'my_model_weights.ckpt')
    
    # Evaluation
    # Get pred sequence
    def get_eval_preds():
        s = trans.model.predict([eval_X, eval_time, eval_states, eval_masks],batch_size=128).argmax(-1)
        idx = 0
        seq = []
        seqs = []
        all_seq = []
        all_true = []

        for item in s:
            if len(seq) == eval_lens[len(seqs)]-1:
                all_seq = all_seq + seq
                all_true = all_true + list(true_states_eval[len(seqs)][1:])
                seqs.append(seq)
                seq = []
                idx = 0
            else:
                seq.append(item[idx+1])
                idx += 1

        all_true = np.array(all_true)
        all_seq = np.array(all_seq)

        return all_true, all_seq
    
    # log likelihood
    def evaluate_likelyhood(all_seq):
        X_ = padd_data(X_observations_eval, trans.len_limit)[:,1:].reshape(-1, 10)
        mask = X_.mean(axis=-1) != 0
        X_ = X_[mask]
        pred = np_utils.to_categorical(all_seq, num_classes= trans.num_states)

        lks_  = np.array([multivariate_normal.logpdf(X_, trans.state_means[k], trans.state_covars[k]).reshape((-1,1)) * pred[:,k].reshape((-1,1)) for k in range(trans.num_states)])
        lks = np.mean(lks_)

        return lks


    all_true, all_seq = get_eval_preds()
    all_true_ = label_exchange(all_true, all_seq, [-10,0,10], trans.state_means.mean(axis=-1))

    acc = (all_true == all_seq).mean()
    f1 = f1_score(all_true_, all_seq, average='macro')
    lks = evaluate_likelyhood(all_seq)

    print('Transformer Model')
    print('Acc :', acc)
    print('F1 Score :', f1)
    print('Log likelyhood :', f1)
    print('')
    return acc, f1, lks



def train_evaluate_att_model(train_data, eval_data):

    tf.reset_default_graph()

    # Get data
    X_observations, true_states, X_time, total_visit = train_data
    X_observations_eval, true_states_eval, X_time_eval, total_visit_eval = eval_data

    # Get model
    model = attentive_state_space_model(num_states=3,
                                maximum_seq_length=25, 
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
                                num_out_layers=1,
                                verbosity = False)

    # Train
    model.fit(X_observations)

    # Evaluate
    def get_eval_preds():
        state_inference, expected_observations, attention = model.predict(X_observations_eval) # Here states has delay

        all_true = []
        all_seq = []
        seqs = []
        for index, item in enumerate(state_inference):
            seq = item.argmax(-1)
            seqs.append(seq)
            all_true = all_true + list(true_states_eval[index][1:]) # delay
            all_seq = all_seq + list(seq[:-1]) # no last one

        all_true = np.array(all_true)
        all_seq = np.array(all_seq)
    
        return all_true, all_seq

    def evaluate_likelyhood(all_seq):
        X_ = padd_data(X_observations_eval, model.maximum_seq_length)[:,1:].reshape(-1, 10) # X has a delay
        mask = X_.mean(axis=-1) != 0
        X_ = X_[mask]
        pred = np_utils.to_categorical(all_seq, num_classes=model.num_states)

        lks_  = np.array([multivariate_normal.logpdf(X_, model.state_means[k], model.state_covars[k]).reshape((-1,1)) * pred[:,k].reshape((-1,1)) for k in range(model.num_states)])
        lks = np.mean(lks_)

        return lks
    
    all_true, all_seq = get_eval_preds()
    all_true_ = label_exchange(all_true, all_seq, [-10,0,10], model.state_means.mean(axis=-1))

    acc = (all_true == all_seq).mean()
    f1 = f1_score(all_true_, all_seq, average='macro')
    lks = evaluate_likelyhood(all_seq)

    print('Attentive Model')
    print('Acc :', acc)
    print('F1 Score :', f1)
    print('Log likelyhood :', f1)
    print('')
    return acc, f1, lks



def main():
    proportion = 0.5

    train_data = generate_data(proportion)
    eval_data = generate_data(proportion)

    print('*'*40)
    print('Design proportion', proportion)
    print('Train proportion', train_data[-1] / 30*1000)
    print('Test proportion', eval_data[-1] / 30*1000)
    print('')

    trans_scores = train_evaluate_my_model(train_data, eval_data)
    att_scores = train_evaluate_my_model(train_data, eval_data)
    
    print('*'*40)





