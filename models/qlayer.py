'''
Code for paper: DEFENSIVE QUANTIZATION LAYER FOR CONVOLUTIONAL NETWORKS AGAINST ADVERSARIAL ATTACK.
By: Sirui Song et.al.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Lambda,Dense,Dropout, Conv2D,Conv1D, ZeroPadding2D, ReLU, MaxPooling2D, MaxPooling1D, Flatten, GlobalAveragePooling2D, Maximum
from keras.layers import Layer,RNN, LSTM, TimeDistributed, Activation, pooling, Reshape, UpSampling2D, Input, Softmax
from keras.layers.normalization import BatchNormalization
from keras.activations import relu, softmax
from keras.models import Model
from keras.metrics import MSE, categorical_crossentropy
import random
from tqdm import tqdm
from six.moves import range
import os
from tensorflow.contrib import losses
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import regularizers

class QLayer(Layer):
    def __init__(self, set_fixed = 0, set_trans = 0, set_att = 8, dense_dim = 32, concept_dim = 32, num_concept = 128, batch_size = 8, distribute_size = 64, regularizer = 0.0005, direct_pass = False):
        self.set_trans = set_trans
        self.set_att = set_att
        self.set_fixed = set_fixed
        self.dense_dim = dense_dim
        self.concept_dim = concept_dim
        self.num_concept = num_concept
        self.num_space = max(self.set_att, self.set_trans, self.set_fixed)

        if self.set_att != 0:
            self.out_dim = concept_dim
            
        elif self.num_space != 0:
            self.out_dim = concept_dim * self.num_space 

        else:
            self.out_dim = concept_dim

        self.batch_size = batch_size
        self.regularizer = regularizer
        self.direct_pass = direct_pass # Direct Pass, NOT USED!
        self.pass_ratio = 0.2
        super(QLayer, self).__init__()

    def build(self, input_shape):
        self.trans = []
        self.embeds = []
        self.sub_embeds = []

        if (self.set_att == 0) and (self.set_trans == 0) and (self.set_fixed == 0):
            print('Do not use projection')
            self.embeds.append(tf.get_variable('embed_'+str(0), [self.num_concept,self.concept_dim], regularizer=regularizers.l2(self.regularizer)))
       
        else:

            if self.set_fixed == 0:

                for i in range(self.num_space):
                    trans = Sequential()
                    trans.add(Dense(self.dense_dim,kernel_regularizer=regularizers.l2(self.regularizer)))
                    trans.add(BatchNormalization())

                    if self.set_trans == 0:
                        trans.add(Activation(activation='softmax'))
                    else:
                        trans.add(Activation(activation='relu'))

                    trans.layers[0].build(input_shape)
                    self.trans.append(trans)
                    self.embeds.append(tf.get_variable('embed_'+str(i), [self.num_concept,self.concept_dim], regularizer=regularizers.l2(self.regularizer)))
            
            else:

                for i in range(self.num_space):
                    self.embeds.append(tf.get_variable('embed_'+str(i), [self.num_concept,self.concept_dim], regularizer=regularizers.l2(self.regularizer)))
        
        if self.direct_pass == True:
            self.pass_threshold = tf.get_variable('pass_threshold', [self.num_space,])

        super(QLayer, self).build(input_shape)
    
    def get_projection(self,x):
        z_ps = []
        shape = x.shape

        if (len(x.shape) > 3):
            x = tf.reshape(x,(-1,x.shape[1]*x.shape[2],x.shape[-1]))

        for i in range(self.num_space):
            z_e_sub = x
            z_e_sub = self.trans[i](z_e_sub)
            z_ps.append(z_e_sub)

        return(z_ps)

    def call(self,x):
        z_qs = []
        z_es = []
        z_ks = []
        z_passes = []
        shape = x.shape

        if (len(x.shape) > 3):
            x = tf.reshape(x,(-1,x.shape[1]*x.shape[2],x.shape[-1]))
            
        num_pass = int(self.pass_ratio * int(x.shape[1]) * self.batch_size)
        
        if self.num_space == 0:
            # No sub-space projection at all
            z_e = x
            z_e_att = x
            _t = tf.expand_dims(z_e_att, axis=-2)
            _e = self.embeds[0]
            _t = tf.norm(_t-_e, axis = -1)
            z_k = tf.argmin(_t,axis=-1) # None * n
            z_q = tf.gather(self.embeds[0],z_k)
            z_k = tf.reshape(z_k, shape = (1,-1,x.shape[1]))

            if (len(shape) > 3):
                z_q = tf.reshape(z_q, shape = (-1, shape[1], shape[2],self.concept_dim))
                z_e = tf.reshape(z_e, shape = (-1, shape[1], shape[2],self.concept_dim))
            else:
                z_q = tf.reshape(z_q, shape = (-1, shape[1],self.concept_dim))
                z_e = tf.reshape(z_e, shape = (-1, shape[1],self.concept_dim))
        
        else:
            for i in range(self.num_space):

                if self.set_fixed == 0:
                    z_e_sub = self.trans[i](x)
                    if self.set_trans == 0:
                        z_e_att = tf.multiply(x, z_e_sub) # Use multiply to perform attention
                    else:
                        z_e_att = z_e_sub
                else:
                    z_e_att = x[:,:,i*self.concept_dim:i*self.concept_dim + self.concept_dim]
                
                z_es.append(z_e_att)
                # Start quantization
                _t = tf.expand_dims(z_e_att, axis=-2)
                _e = self.embeds[i]
                _t = tf.norm(_t-_e, axis = -1)
                z_k = tf.argmin(_t,axis=-1) # None * n
                _current_qs = tf.gather(self.embeds[i],z_k)
                # end quantization

                if self.direct_pass == True:
                
                    _dis = tf.reduce_min(_t,axis = -1)
                    _pass_threshold = tf.sort(tf.reshape(_dis,(-1,)))[-num_pass]  
                    _pass = tf.greater(tf.reduce_min(_t,axis=-1), _pass_threshold)
                    z_passes.append(_pass)
                    _pass = tf.tile(tf.expand_dims(_pass,-1),[1,1,self.dense_dim])
                    _current_qs = tf.cast(_pass, tf.float32) * z_e_att + tf.cast(tf.logical_not(_pass),tf.float32) * _current_qs
                
                z_ks.append(tf.reshape(z_k, shape = (1,-1,x.shape[1])))
                z_qs.append(_current_qs)

            if self.set_att == 0:
                z_q = tf.concat(z_qs, axis=-1)
                z_e = tf.concat(z_es, axis=-1)
                z_k = tf.concat(z_ks, axis=0)
            else:
                z_q = tf.add_n(z_qs)
                z_e = tf.add_n(z_es)
                z_k = tf.concat(z_ks, axis=0)

            if (len(shape) > 3):
                z_q = tf.reshape(z_q, shape = (-1, shape[1], shape[2],self.out_dim))
                z_e = tf.reshape(z_e, shape = (-1, shape[1], shape[2],self.out_dim))
            else:
                z_q = tf.reshape(z_q, shape = (-1, shape[1],self.out_dim))
                z_e = tf.reshape(z_e, shape = (-1, shape[1],self.out_dim))

        if self.direct_pass:
            z_pass = tf.concat(z_passes, axis = 0)
            return([z_q, z_e, z_k,z_pass])
            
        else:
            return([z_q, z_e, z_k])
    
    def compute_output_shape(self,input_shape):
        shape = input_shape

        if (len(input_shape) > 3):
            s = (None, int(shape[1]), int(shape[2]),self.out_dim)
            s2 =  (None, self.num_space, int(shape[1]) * int(shape[2]))

        else:
            s = (None, int(shape[1]), self.out_dim)
            s2 =  (None, self.num_space, int(shape[1]))

        if self.direct_pass:
            return([s,s,s2,s2])
        else:
            return([s,s,s2])

    # def call_with_actives(self,x,actives):
    #     z_qs = []
    #     z_es = []
    #     z_ks = []

    #     for i in range(self.num_space):
    #         z_e_sub = x
    #         trans = self.trans[i]

    #         for item in trans:
    #             z_e_sub = item(z_e_sub)

    #         if self.set_trans:
    #             z_es.append(z_e_sub)

    #         elif self.set_att:
    #             z_e_att = tf.multiply(x, z_e_sub) # Use multiply to perform attention
    #             z_es.append(z_e_att)

    #         _t = tf.expand_dims(z_es[i], axis=-2)
    #         _e = tf.boolean_mask(self.embeds[i], actives[i], axis=0) # active * concept_dim
    #         _t = tf.norm(_t-_e, axis = -1)
    #         z_k = tf.argmin(_t,axis=-1) 
    #         z_ks.append(tf.reshape(z_k, shape = (1,-1,z_e_att.shape[1],self.num_concept)))
    #         z_qs.append(tf.gather(_e,z_k))

    #     z_q = tf.concat(z_qs, axis=-1)
    #     z_e = tf.concat(z_es, axis=-1)

    #     if (len(shape) > 3):
    #         z_q = tf.reshape(z_q, shape = (-1, shape[1], shape[2],self.out_dim))
    #         z_e = tf.reshape(z_e, shape = (-1, shape[1], shape[2],self.out_dim))
    #     else:
    #         z_q = tf.reshape(z_q, shape = (-1, shape[1],self.out_dim))
    #         z_e = tf.reshape(z_e, shape = (-1, shape[1],self.out_dim))

    #     return(z_q, z_e, z_ks)
    

