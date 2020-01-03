### Setting up the experiment 
from data.make_data import generate_trajectory_new, generate_trajectory
# from keras import optimizers, losses
# import keras.backend as K
import numpy as np
from utils import padd_data
import os
from models.my_model import *
from keras_radam import RAdam
from sklearn.metrics import f1_score

'''Generate data'''
X_observations, true_states, X_time = generate_trajectory_new(num_states=3, 
                                                    Num_observations=10, 
                                                    Num_samples=1000, 
                                                    Max_seq=25, 
                                                    Min_seq=5, 
                                                    Max_length=30,
                                                    alpha=100,
                                                    P_trans = np.array([[0.9, 0.1, 0.06], 
                                                                        [0.6, 0.1, 0.3], 
                                                                        [0.1, 0.8, 0.1]]),
                                                    reverse_mode=False)

np.save('./data/syn_data_new.npy',{'X':X_observations, 'states':true_states, 'time':X_time})

data = np.load('./data/syn_data_new.npy', allow_pickle = True).item()
X_observations = data['X']
true_states = data['states']
# X_time = data['time']



'''Model'''
trans = TranModel(len_limit=25, time_limit=30, d_data=10, d_model=32, d_inner_hid=32, n_head=2, layers=2, num_states=3)
X_padded, states_padded = trans.prepare_data(X_observations)
# X_time = np.repeat(np.expand_dims(np.arange(30), axis=0),len(X_padded), axis=0)

mask = X_padded[:,:].mean(axis=-1) != 0
lens = mask.sum(axis=-1)

count = []
for i in range(3):
    count.append(len(np.where(states_padded[:,:,i]==1)[0]))

count = np.array(count)
count = count.sum() / count

'''Prepare data'''
train_X = []
train_states = []
train_masks = []
train_time = []

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



'''Build model'''
count = np.array([1,1,1]).reshape(-1,1)
trans.build_transformer_model(class_weights = count.reshape(-1,1))

trans.model.compile(optimizer=RAdam(1e-3))
# trans.model.load_weights('./save/my_model_weights.ckpt')

trans.model.fit([train_X, train_time, train_states, train_masks], batch_size=100, epochs=500, validation_split=0.5)
# trans.model.save_weights('./save/my_model_weights.ckpt')


'''Validation'''
s = trans.model.predict([train_X, train_time, train_states, train_masks],batch_size=128) 
s = s.argmax(-1)

idx = 1
seq = [0]
seqs = []
all_seq = []
all_true = []
corr = 0


# Inference mode 1: predict one by one (prefered, as we do not add the history prediction into loss)
for item in s:
    if len(seq) == lens[len(seqs)]:
        corr += np.array(seq == list(true_states[len(seqs)])).sum()
        all_seq = all_seq + seq
        all_true = all_true + list(true_states[len(seqs)])
        seqs.append(seq)
        seq = [0]
        idx = 1
    else:
        seq.append(item[idx])
        idx += 1

print('Inference mode 1, Acc :',(np.array(all_true) == np.array(all_seq)).mean())
print('F1 Score :',f1_score(all_true, all_seq, average='micro'))



# Inference mode 2: predict one by one (prefered, as we do not add the history prediction into loss)
idx = 0
seq = []
seqs = []
all_seq = []
all_true = []
corr = 0

for l in lens:
    seq = s[idx+l-1]
    idx += l-1
    corr += np.array(seq[:l] == true_states[len(seqs)]).sum()
    all_seq = all_seq + list(seq[:l])
    all_true = all_true + list(true_states[len(seqs)])
    seqs.append(seq)


print('Acc :',corr / sum(lens))
print('F1 Score :',f1_score(all_true, all_seq, average='macro'))



