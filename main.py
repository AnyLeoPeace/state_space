### Setting up the experiment 
from data.make_data import generate_trajectory_new, generate_trajectory
# from keras import optimizers, losses
# import keras.backend as K
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
max_seq = 15
min_seq = 5
max_length = 30
alpha = 100

X_observations, true_states, X_time = generate_trajectory_new(num_states=3, 
                                                    Num_observations=10, 
                                                    Num_samples=1000, 
                                                    Max_seq=max_seq, 
                                                    Min_seq=min_seq, 
                                                    Max_length=max_length,
                                                    alpha=alpha,
                                                    P_trans = np.array([[0.85, 0.1, 0.05], 
                                                                        [0.1, 0.7, 0.2], 
                                                                        [0, 0.2, 0.8]]),
                                                    reverse_mode=False)

np.save('./data/syn_data_eval' + str(max_seq) + '_'  + str(min_seq) + '_' + str(max_length) + '_' + str(alpha),{'X':X_observations, 'states':true_states, 'time':X_time})


data = np.load('./data/syn_data_' + str(max_seq) + '_'  + str(min_seq) + '_' + str(max_length) + '_' + str(alpha) + '.npy', allow_pickle = True).item()
X_observations = data['X']
true_states = data['states']
X_time = data['time']



'''Model'''
trans = TranModel(len_limit=25, time_limit=30, d_data=10, d_model=16, d_inner_hid=16, n_head=2, layers=2, num_states=3, dropout=0.3)
X_padded, states_padded = trans.prepare_data(X_observations) # Here states has no delay
# X_time = np.repeat(np.expand_dims(np.arange(30), axis=0),len(X_padded), axis=0)

mask = X_padded[:,:].mean(axis=-1) != 0
lens = mask.sum(axis=-1)

count = []
for i in range(3):
    count.append(len(np.where(states_padded[:,:,i]==1)[0]))

count = np.array(count)
count = count.sum() / count

'''Prepare data'''
def get_train_data(X_padded, states_padded, X_time):
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
                times = list(times) + [trans.time_limit-1] * (trans.len_limit-len(times))
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


train_X, train_time, train_states, train_masks = get_train_data(X_padded, states_padded, X_time)


'''Build model'''
count = np.array([1,1,1]).reshape(-1,1)
trans.build_transformer_model(class_weights = count.reshape(-1,1))

trans.model.compile(optimizer=RAdam(1e-3))
# trans.model.load_weights('./save/my_model_weights.ckpt')


'''Train'''

# Val set
data_eval = np.load('./data/syn_data_eval' + str(max_seq) + '_'  + str(min_seq) + '_' + str(max_length) + '_' + str(alpha) + '.npy', allow_pickle = True).item()
X_observations_eval = data_eval['X']
true_states_eval = data_eval['states']
X_time_eval = data_eval['time']

X_padded_eval,states_padded_eval = trans.prepare_data(X_observations_eval, training = False)
eval_X, eval_time, eval_states, eval_masks = get_train_data(X_padded_eval, states_padded_eval, X_time_eval)


his = trans.model.fit([train_X, train_time, train_states, train_masks], batch_size=100, epochs=100, verbose=2)

# for i in range(0,100):
    # his = trans.model.fit([train_X, train_time, train_states, train_masks], batch_size=100, epochs=i+1, initial_epoch = i, verbose=2)
    # acc = trans.model.evaluate([eval_X, eval_time, eval_states, eval_masks],batch_size=100, verbose = 0)[-1]
    # print('Val Acc :', acc)


trans.model.save_weights('./save/my_model_weights.ckpt')


'''Evaluation'''

mask = X_padded_eval[:,:].mean(axis=-1) != 0
eval_lens = mask.sum(axis=-1)

acc = trans.model.evaluate([eval_X, eval_time, eval_states, eval_masks],batch_size=128, verbose=0)[-1]
print('Acc :', acc)


# Log likelyhood
s = trans.model.predict([eval_X, eval_time, eval_states, eval_masks],batch_size=128) 
s = s.argmax(-1)

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

print('Acc :',(all_true == all_seq).mean())
print('Acc :',(label_exchange(0,2,all_true) == all_seq).mean())
print('F1 Score :',f1_score(all_true, all_seq, average='macro'))


# log likelihood
X_ = padd_data(X_observations_eval, trans.len_limit)[:,1:].reshape(-1, 10)
mask = X_.mean(axis=-1) != 0
X_ = X_[mask]
pred = np_utils.to_categorical(all_seq, num_classes=trans.num_states)

lks_  = np.array([multivariate_normal.logpdf(X_, trans.state_means[k], trans.state_covars[k]).reshape((-1,1)) * pred[:,k].reshape((-1,1)) for k in range(trans.num_states)])
lks = np.mean(lks_)

print('likelyhodd',lks)



