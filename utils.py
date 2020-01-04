import keras.backend as K
from keras.layers import Layer
# from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow as tf
import numpy as np
from scipy.spatial import distance_matrix

def padd_data(X, padd_length):
    
    X_padded      = []
    
    for k in range(len(X)):
        
        if X[k].shape[0] < padd_length:
            
            if len(X[k].shape) > 1:
                X_padded.append(np.array(np.vstack((np.array(X[k]), 
                                                    np.zeros((padd_length-X[k].shape[0],X[k].shape[1]))
                                                    ))))
            else:
                X_padded.append(np.array(np.vstack((np.array(X[k]).reshape((-1,1)),
                                                    np.zeros((padd_length-X[k].shape[0],1))
                                                    ))))
                
        else:
            
            if len(X[k].shape) > 1:
                X_padded.append(np.array(X[k]))
            else:
                X_padded.append(np.array(X[k]).reshape((-1,1)))
  

    X_padded      = np.array(X_padded)

    return X_padded


def label_exchange(labels, preds, labels_mean, preds_mean, num_states = 3):
    '''
    Input the labels sequence and the pred sequence
    Exchange the labels alignment to maximize the acc
    '''
    dis = distance_matrix(labels_mean.mean(axis=-1), preds_mean.mean(axis=-1))
    pos_a,pos_b = np.where(abs(dis) < 2)
    new_labels = np.copy(labels)

    for i in range(num_states):
        a = pos_a[i]
        b = pos_b[i]
        ma = labels == a
        new_labels[ma] = b

    return new_labels