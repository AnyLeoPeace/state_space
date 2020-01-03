from keras import regularizers
from keras.models import Model
from keras import backend as K
from keras.layers import *
from models.SeqModels import *
from models.original_transformer import *
from keras_transformer.transformer import *

class TranModel():

    def __init__(self, len_limit = 50, time_limit = 50, \
              d_data = 16, d_model=64, d_inner_hid=64, \
              n_head=4, layers=2, dropout=0.1, \
              num_states = 3):
        '''
        num_hidden_trans_layer: how many layers to transform ht to xt
        '''

        self.len_limit = len_limit
        self.time_limit = time_limit
        self.d_model = d_model
        self.d_data = d_data
        self.layers = layers
        self.num_states = num_states
        self.dropout = dropout
        self.n_head = n_head

    def build_transformer_model(self, class_weights, confidence_penalty_weight = 0.1):
        class_weights = tf.convert_to_tensor(class_weights, dtype='float32')
        
        # Input
        input_seq = Input(shape=(self.len_limit, self.d_data))
        input_time = Input(shape=(self.len_limit, ))
        input_states = Input(shape=(self.len_limit, self.num_states), dtype='int32')
        input_predict_mask = Input(shape=(self.len_limit, ), dtype='bool') # Where to predict
        seq_mask = Lambda(lambda x:tf.cast(tf.reduce_any(K.not_equal(x, 0), axis=-1), 'bool'), name = 'seq_mask')(input_seq)

        # Embedding
        embed_layer = TimeDistributed(Dense(self.d_model,name = 'emb_dense'))
        pos_emb = PosEncodingLayer(self.time_limit, self.d_model)
        # pos_emb = Embedding(self.time_limit, self.d_model)
        unk_emb_layer = Embedding(2, self.d_model)

        # Covert future visits to <UNKNOWN>
        def convert(inputs):
            seq, mask = inputs
            unk_emb = unk_emb_layer(tf.cast(mask,'int32'))
            seq = tf.where(tf.transpose(K.repeat(mask,self.d_model), perm=[0,2,1]), unk_emb, seq)
            return seq            

        unk_convert_layer = Lambda(convert)

        def inverse_norm(x):
            return (x * self.normalizer.scale_) + (x * self.normalizer.mean_)

        
        # Output
        output_softmax_layer = Softmax(name='prediction')
        dense_layer = TimeDistributed(Dense(self.num_states,name = 'dense'))


        # Build
        next_step_input = embed_layer(input_seq)
        next_step_input = unk_convert_layer([next_step_input, input_predict_mask])

        next_step_input = add_layer([next_step_input, pos_emb(input_time, pos_input = True)])
        # next_step_input = Concatenate()([next_step_input, pos_emb(input_time, pos_input = True)])

        for i in range(self.layers):
            next_step_input = TransformerBlock(
                                name='transformer' + str(i), num_heads=self.n_head,
                                residual_dropout=self.dropout,
                                attention_dropout=self.dropout,
                                use_masking=False,
                                vanilla_wiring=True)(next_step_input)
        
        logits = dense_layer(next_step_input)
        pred = output_softmax_layer(logits)


        model = Model(inputs=[input_seq, input_time, input_states, input_predict_mask], outputs=pred)

        # mask = tf.math.logical_or(input_predict_mask, seq_mask)
        mask = input_predict_mask
        mask = tf.cast(mask, 'float32')
        loss = tf.cast(input_states, 'float32') * tf.log(pred + 1e-7)

        # loss = K.dot(loss, class_weights)
        loss = -tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
        loss = K.mean(loss)

        # Metric Acc
        corr = K.cast(K.equal(K.cast(K.argmax(pred,axis=-1), 'int32'), K.cast(K.argmax(input_states, axis=-1),'int32')), 'float32')
        corr = K.sum(corr * mask, -1) / K.sum(mask, -1)


        confidence_penalty = confidence_penalty_weight * K.sum(pred * K.log(pred), axis=-1)
        confidence_penalty = tf.reduce_sum(confidence_penalty * mask, -1) / tf.reduce_sum(mask, -1)
        confidence_penalty = K.mean(confidence_penalty)

        model.add_loss(confidence_penalty)
        model.add_loss(loss)

        model.add_metric(corr, 'predict_acc')

        self.model = model
        return model


    def initialize_hidden_states(self, X):
        
        self.init_states = GaussianMixture(n_components=self.num_states, 
                                           covariance_type='full')
        
        self.init_states.fit(np.concatenate(X).reshape((-1, self.d_data))) # Similar to KMeans init


    def prepare_data(self,X, training = True):
        
        self.state_trajectories_ = []
        
        if training:
            
            self.initialize_hidden_states(X)
            self.normalizer   = StandardScaler()
            self.normalizer.fit(np.concatenate(X))
        
        # Initial predict using GM
        state_inferences_init  = [np.argmax(self.init_states.predict_proba(X[k]), axis=1) for k in range(len(X))]
        self.all_states        = state_inferences_init

        for v in range(len(state_inferences_init)):
            
            state_list = [state_to_array(state_inferences_init[v][k], self.num_states) for k in range(len(state_inferences_init[v]))] # Convert label to one-hot array
            delayed_traject = np.vstack((np.array(state_list)[1:, :], np.array(state_list)[-1, :]))
            traject = np.array(state_list)
            '''NOTE: Here I do not use delayde traj'''
            self.state_trajectories_.append(traject)
            

        self.X_normalized  = []

        for k in range(len(X)):
            self.X_normalized.append(self.normalizer.transform(X[k])) 

        self.X_, self.state_update = padd_data(self.X_normalized, self.len_limit), padd_data(self.state_trajectories_, self.len_limit)
        

        if training:

            # Baseline transition matrix
            # Calculate the initial transition probability
            initial_states = np.array([self.all_states[k][0] for k in range(len(self.all_states))]) # the first state for each sample
            init_probs     = [np.where(initial_states==k)[0].shape[0] / len(initial_states) for k in range(self.num_states)] # prior distribution for states

            transits   = np.zeros((self.num_states, self.num_states))
            each_state = np.zeros(self.num_states)
            
            for _ in range(len(self.all_states)):
            
                new_trans, new_each_state = get_transitions(self.all_states[_], self.num_states)
        
                transits   += new_trans
                each_state += new_each_state
        
            for _ in range(self.num_states):
        
                transits[_, :] = transits[_, :] / each_state[_]
                transits[_, :] = transits[_, :] / np.sum(transits[_, :])
    
            self.initial_probabilities = np.array(init_probs)
            self.transition_matrix     = np.array(transits)
            
            # -----------------------------------------------------------
            # Observational distribution
            # -----------------------------------------------------------
            
            self.state_means  = self.init_states.means_
            self.state_covars = self.init_states.covariances_   

        return self.X_, self.state_update 

        