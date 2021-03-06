### Setting up the experiment
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from data.make_data import *
import numpy as np
from utils import *
import os
from models.my_model import *
from keras_radam import RAdam
from sklearn.metrics import f1_score
from scipy.stats import multivariate_normal
from keras.utils import np_utils

import logging
import time

localtime = time.asctime( time.localtime(time.time()) )

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= localtime + '_log',
                filemode='w')


console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

data_mean = np.array([-10,0,10])
train_epoch = [100,100]
verbose = False


'''Generate data'''
def generate_data_mode_1(max_seq = 25, min_seq = 5, max_length = 50, alpha = 100, proportion = 0.5, personalized = 0):

    X_observations, true_states, X_time, total_visit = generate_trajectory_final(num_states=3, 
                                                        Num_observations=10, 
                                                        Num_samples=1000, 
                                                        Max_seq=max_seq, 
                                                        Min_seq=min_seq, 
                                                        Max_length=max_length,
                                                        alpha=alpha,
                                                        proportion = proportion,
                                                        personalized = personalized,
                                                        P_trans = np.array([[0.85, 0.1, 0.05], 
                                                                            [0.1, 0.7, 0.2], 
                                                                            [0, 0.2, 0.8]]),
                                                        mu_ = data_mean,
                                                        reverse_mode=False)
    
    return X_observations, true_states, X_time, total_visit


def generate_data_mode_2(max_seq = 20, min_seq = 10, max_length = 30, alpha = 100):

    X_observations, true_states, X_time, total_visit = generate_trajectory_new(num_states=3, 
                                                        Num_observations=10, 
                                                        Num_samples=1000, 
                                                        Max_seq=max_seq, 
                                                        Min_seq=min_seq, 
                                                        Max_length=max_length,
                                                        alpha=alpha,
                                                        P_trans = np.array([[0.85, 0.1, 0.05], 
                                                                            [0.1, 0.7, 0.2], 
                                                                            [0, 0.2, 0.8]]),
                                                        mu_ = data_mean,
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





def train_evaluate_my_model(train_data, eval_data, save = None, personalized = False, time_limit=30, len_limit=25):
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

    train_his_mask = X_padded[:,:].mean(axis=-1) != 0
    lens = train_his_mask.sum(axis=-1)

    eval_his_mask = X_padded_eval[:,:].mean(axis=-1) != 0
    eval_lens = eval_his_mask.sum(axis=-1)

    # Train
    his = trans.model.fit([train_X, train_time, train_states, train_masks], batch_size=100, epochs=train_epoch[0], verbose=verbose)

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

    def evaluate_likelyhood_with_personalization(all_seq):
        # A simplified personalized effect
        # The predicted dif is the average of history dif
        # Not completed
        eval_X_ = trans.normalizer.inverse_transform(eval_X)
        
        history_mu = np.take(trans.state_means, np.argmax(eval_states, axis=-1), axis=0) * np.expand_dims(eval_states.mean(axis= -1) != 0, axis=-1)
        history_dif_ = eval_X - history_mu
        history_dif = history_dif_.sum(axis=1) / np.expand_dims((eval_X.mean(axis=-1) !=0).sum(axis=1), axis = -1)

        # remove the last prediction
        pred_dif = []
        for index, item in enumerate(history_dif): 
            l = int(eval_states[index].sum(axis = -1).sum())
            current_l = (eval_X[index].mean(-1) != 0).sum()
            if current_l < l:
                # This is not the last postion in the sequence
                pred_dif.append(item)

        pred_dif = np.array(pred_dif)
        X_ = padd_data(X_observations_eval, trans.len_limit)[:,1:]
        mask = X_.mean(axis=-1) != 0
        X_ = X_[mask]
        X_ = X_ - pred_dif

        pred = np_utils.to_categorical(all_seq, num_classes= trans.num_states)

        lks_  = np.array([multivariate_normal.logpdf(X_, trans.state_means[k], trans.state_covars[k]).reshape((-1,1)) * pred[:,k].reshape((-1,1)) for k in range(trans.num_states)])
        lks = np.mean(lks_)

        return lks


    all_true, all_seq = get_eval_preds()
    all_true_ = label_exchange(all_true, all_seq, data_mean, trans.state_means.mean(axis=-1))

    acc = (all_true_ == all_seq).mean()
    f1 = f1_score(all_true_, all_seq, average='macro')
    lks = evaluate_likelyhood(all_seq)

    if personalized:
        lks_p = evaluate_likelyhood_with_personalization(all_seq)

    logging.info('Transformer Model')
    logging.info('Acc : %2.4f'% acc)
    logging.info('F1 Score : %2.4f'% f1)
    logging.info('Log likelyhood : %2.4f'%lks)

    if personalized:
        logging.info('Log likelyhood with personalization: %2.4f'%lks_p)
        logging.info('')

        return acc, f1, lks, lks_p
    
    else:
        logging.info('')

        return acc, f1, lks





def train_evaluate_att_model(train_data, eval_data, len_limit = 25):


    # Get data
    X_observations, true_states, X_time, total_visit = train_data
    X_observations_eval, true_states_eval, X_time_eval, total_visit_eval = eval_data

    # Get model
    model = attentive_state_space_model(num_states=3,
                                maximum_seq_length=len_limit, 
                                input_dim=10, 
                                rnn_type='LSTM',
                                latent=True,
                                generative=True,
                                num_iterations=50, # Not used
                                num_epochs=train_epoch[1], 
                                batch_size=100, 
                                learning_rate=1e-3, 
                                num_rnn_hidden=16, # Keep the same with the transformer model
                                num_rnn_layers=1,
                                dropout_keep_prob=0.1,
                                num_out_hidden=16, 
                                num_out_layers=1,
                                verbosity = verbose)

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
    all_true_ = label_exchange(all_true, all_seq, data_mean, model.state_means.mean(axis=-1))

    acc = (all_true_ == all_seq).mean()
    f1 = f1_score(all_true_, all_seq, average='macro')
    lks = evaluate_likelyhood(all_seq)

    logging.info('Attentive Model')
    logging.info('Acc : %2.4f'% acc)
    logging.info('F1 Score : %2.4f'% f1)
    logging.info('Log likelyhood : %2.4f'%lks)
    logging.info('')


    return acc, f1, lks



def main():

    logging.info('Begining Comparing')

    data_config = {
        'max_length':50,
        'min_seq':5,
        'max_seq':25,
    }

    for proportion in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:

        train_data = generate_data_mode_1(proportion=proportion, max_length=data_config['max_length'], min_seq=data_config['min_seq'], max_seq=data_config['max_seq'])
        eval_data = generate_data_mode_1(proportion=proportion, max_length=data_config['max_length'], min_seq=data_config['min_seq'], max_seq=data_config['max_seq'])

        total = data_config['max_length']*1000

        logging.info('*'*40)
        logging.info('Design proportion : %2.4f'% proportion)
        logging.info('Train proportion : %2.4f'% (train_data[-1] / total))
        logging.info('Test proportion : %2.4f'% (eval_data[-1] / total))
        logging.info('')

        for i in range(8):

            logging.info('Compare Iteration %d' % (i+1)+'-'*20)
            logging.info('')

            trans_scores = train_evaluate_my_model(train_data, eval_data, personalized=False, time_limit=data_config['max_length'], len_limit = data_config['max_seq'])
            att_scores = train_evaluate_att_model(train_data, eval_data, len_limit = data_config['max_seq'])

            np.save('./results/proportion_' + str(proportion) + '_iteration_'+str(i),{'trans':trans_scores, 'att':att_scores, 'proportion':(train_data[-1] / total, eval_data[-1] / total)})
        
        logging.info('*'*40)


if __name__ == '__main__':
    main()


