import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
tfe = tf.contrib.eager
import tensorflow_probability as tfp
from keras.optimizers import RAdam, SGD
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.utils import np_utils
import keras.regularizers as regularizers
from keras_radam import RAdam
try:
	from tqdm import tqdm
	from dataloader import TokenList, pad_to_longest
	# for transformer
except: pass

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
	def __init__(self, attn_dropout=0.1):
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):   # mask_k or mask_qk
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/temper)([q, k])  # shape=(batch, q, k)
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, dropout, mode=0):
		self.mode = mode
		self.n_head = n_head
		self.d_k = self.d_v = d_k = d_v = d_model // n_head
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention()
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
				x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
				return x
			qs = Lambda(reshape1)(qs)
			ks = Lambda(reshape1)(ks)
			vs = Lambda(reshape1)(vs)

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
				return x
			head = Lambda(reshape2)(head)
		elif self.mode == 1:
			heads = []; attns = []
			for i in range(n_head):
				qs = self.qs_layers[i](q)   
				ks = self.ks_layers[i](k) 
				vs = self.vs_layers[i](v) 
				head, attn = self.attention(qs, ks, vs, mask)
				heads.append(head); attns.append(attn)
			head = Concatenate()(heads) if n_head > 1 else heads[0]
			attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		return outputs, attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer = LayerNormalization()
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.norm_layer(Add()([enc_input, output]))
		output = self.pos_ffn_layer(output)
		return output, slf_attn

class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.enc_att_layer  = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer1 = LayerNormalization()
		self.norm_layer2 = LayerNormalization()
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
		if dec_last_state is None: dec_last_state = dec_input
		output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
		x = self.norm_layer1(Add()([dec_input, output]))
		output, enc_attn = self.enc_att_layer(x, enc_output, enc_output, mask=enc_mask)
		x = self.norm_layer2(Add()([x, output]))
		output = self.pos_ffn_layer(x)
		return output, slf_attn, enc_attn

def GetPosEncodingMatrix(max_len, d_emb, period = 10000):
	pos_enc = np.array([
		[pos / np.power(period, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc


def GetPadMask(q, k):
	'''
	shape: [B, Q, K]
	'''
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	'''
	shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
	'''
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask

class SelfAttention():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
		if return_att: atts = []
		mask = Lambda(lambda x:K.sum(K.cast(K.greater(x, 0), 'float32'), axis=-1))(src_seq)
		x = src_emb		
		for enc_layer in self.layers[:active_layers]:
			x, att = enc_layer(x, mask)
			if return_att: atts.append(att)
		return (x, atts) if return_att else x


class MaskedSelfAttention():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
		if return_att: atts = []
		self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(src_seq)
		self_sub_mask = Lambda(GetSubMask)(src_seq)
		self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
		x = src_emb		
		for enc_layer in self.layers[:active_layers]:
			x, att = enc_layer(x, self_mask)
			if return_att: atts.append(att)
		return (x, atts) if return_att else x

class Decoder():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
		x = tgt_emb
		self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(tgt_seq)
		self_sub_mask = Lambda(GetSubMask)(tgt_seq)
		self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
		enc_mask = Lambda(lambda x:GetPadMask(x[0], x[1]))([tgt_seq, src_seq])
		if return_att: self_atts, enc_atts = [], []
		for dec_layer in self.layers[:active_layers]:
			x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
			if return_att: 
				self_atts.append(self_att)
				enc_atts.append(enc_att)
		return (x, self_atts, enc_atts) if return_att else x

class DecoderPerStep(Layer):
	def __init__(self, decoder):
		super().__init__()
		self.layers = decoder.layers
	def call(self, inputs):
		(x, src_seq, enc_output), tgt_embs = inputs[:3], inputs[3:]
		enc_mask = K.cast(K.greater(src_seq, 0), 'float32')
		llen = tf.shape(tgt_embs[0])[1]
		col_mask = K.cast(K.equal(K.cumsum(K.ones_like(tgt_embs[0], dtype='int32'), axis=1), llen), dtype='float32')
		rs = [x]
		for i, dec_layer in enumerate(self.layers):
			tgt_emb = tgt_embs[i] + x * col_mask
			x, _, _ = dec_layer(x, enc_output, enc_mask=enc_mask, dec_last_state=tgt_emb)
			rs.append(x)
		return rs
	def compute_output_shape(self, ishape):
		return [ishape[0] for _ in range(len(self.layers)+1)]

class ReadoutDecoderCell(Layer):
	def __init__(self, o_word_emb, pos_emb, decoder, target_layer, **kwargs):
		self.o_word_emb = o_word_emb
		self.pos_emb = pos_emb
		self.decoder = decoder
		self.target_layer = target_layer
		super().__init__(**kwargs)
	def call(self, inputs, states, constants, training=None):
		(tgt_curr_input, tgt_pos_input, dec_mask), dec_output = states[:3], list(states[3:])
		enc_output, enc_mask = constants

		time = K.max(tgt_pos_input)
		col_mask = K.cast(K.equal(K.cumsum(K.ones_like(dec_mask), axis=1), time), dtype='int32')
		dec_mask = dec_mask + col_mask

		tgt_emb = self.o_word_emb(tgt_curr_input)
		if self.pos_emb: tgt_emb = tgt_emb + self.pos_emb(tgt_pos_input, pos_input=True)

		x = tgt_emb
		xs = []
		cc = K.cast(K.expand_dims(col_mask), dtype='float32')
		for i, dec_layer in enumerate(self.decoder.layers):
			dec_last_state = dec_output[i] * (1-cc) + tf.einsum('ijk,ilj->ilk', x, cc)
			x, _, _ = dec_layer(x, enc_output, dec_mask, enc_mask, dec_last_state=dec_last_state)
			xs.append(dec_last_state)

		ff_output = self.target_layer(x)
		out = K.cast(K.argmax(ff_output, -1), dtype='int32')
		return out, [out, tgt_pos_input+1, dec_mask] + xs

class InferRNN(Layer):
	def __init__(self, cell, return_sequences=False, go_backwards=False, **kwargs):
		if not hasattr(cell, 'call'):
			raise ValueError('`cell` should have a `call` method. ' 'The RNN was passed:', cell)
		super().__init__(**kwargs)
		self.cell = cell
		self.return_sequences = return_sequences
		self.go_backwards = go_backwards

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], 1) if self.return_sequences else (input_shape[0], 1)
			
	def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
		if initial_state is not None:
			kwargs['initial_state'] = initial_state
		if constants is not None:
			kwargs['constants'] = constants
			self._num_constants = len(constants)
		return super().__call__(inputs, **kwargs)

	def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
		if isinstance(inputs, list):
			if self._num_constants is None: initial_state = inputs[1:]
			else: initial_state = inputs[1:-self._num_constants]
			inputs = inputs[0]
		input_shape = K.int_shape(inputs)
		timesteps = input_shape[1]

		kwargs = {}
		def step(inputs, states):
			constants = states[-self._num_constants:]
			states = states[:-self._num_constants]
			return self.cell.call(inputs, states, constants=constants, **kwargs)

		last_output, outputs, states = K.rnn(step, inputs, initial_state, constants=constants,
											 go_backwards=self.go_backwards,
											 mask=mask, unroll=False, input_length=timesteps)
		output = outputs if self.return_sequences else last_output
		return output

def decode_batch_greedy(src_seq, encode_model, decode_model, start_mark, end_mark, max_len=128):
	enc_ret = encode_model.predict_on_batch(src_seq)
	bs = src_seq.shape[0]
	target_one = np.zeros((bs, 1), dtype='int32')
	target_one[:,0] = start_mark
	d_model = decode_model.inputs[-1].shape[-1]
	n_dlayers = len(decode_model.inputs) - 3
	dec_outputs = [np.zeros((bs, 1, d_model)) for _ in range(n_dlayers)]
	ended = [0 for x in range(bs)]
	decoded_indexes = [[] for x in range(bs)]
	for i in range(max_len-1):
		outputs = decode_model.predict_on_batch([target_one, src_seq, enc_ret] + dec_outputs)
		new_dec_outputs, output = outputs[:-1], outputs[-1]
		for dec_output, new_out in zip(dec_outputs, new_dec_outputs): 
			dec_output[:,-1,:] = new_out[:,0,:]
		dec_outputs = [np.concatenate([x, np.zeros_like(new_out)], axis=1) for x in dec_outputs]

		sampled_indexes = np.argmax(output[:,0,:], axis=-1)
		for ii, sampled_index in enumerate(sampled_indexes):
			if sampled_index == end_mark: ended[ii] = 1
			if not ended[ii]: decoded_indexes[ii].append(sampled_index)
		if sum(ended) == bs: break
		target_one[:,0] = sampled_indexes
	return decoded_indexes

def decode_batch_beam_search(src_seq, topk, encode_model, decode_model, start_mark, end_mark, max_len=128, early_stop_mult=5):
	N = src_seq.shape[0]
	src_seq = src_seq.repeat(topk, 0)
	enc_ret = encode_model.predict_on_batch(src_seq)
	bs = src_seq.shape[0]

	target_one = np.zeros((bs, 1), dtype='int32')
	target_one[:,0] = start_mark
	d_model = decode_model.inputs[-1].shape[-1]
	n_dlayers = len(decode_model.inputs) - 3
	dec_outputs = [np.zeros((bs, 1, d_model)) for _ in range(n_dlayers)]

	final_results = []
	decoded_indexes = [[] for x in range(bs)]
	decoded_logps = [0] * bs
	lastks = [1 for x in range(N)]
	bests = {}
	for i in range(max_len-1):
		outputs = decode_model.predict_on_batch([target_one, src_seq, enc_ret] + dec_outputs)
		new_dec_outputs, output = outputs[:-1], outputs[-1]
		for dec_output, new_out in zip(dec_outputs, new_dec_outputs): 
			dec_output[:,-1,:] = new_out[:,0,:]

		dec_outputs = [np.concatenate([x, np.zeros_like(new_out)], axis=1) for x in dec_outputs]

		output = np.exp(output[:,0,:])
		output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)

		next_dec_outputs = [x.copy() for x in dec_outputs]
		next_decoded_indexes = [1 for x in range(bs)]

		for ii in range(N):
			base = ii * topk
			cands = []
			for k, wprobs in zip(range(lastks[ii]), output[base:,:]):
				prev = base+k
				if len(decoded_indexes[prev]) > 0 and decoded_indexes[prev][-1] == end_mark: continue
				ind = np.argpartition(wprobs, -topk)[-topk:]
				wsorted = [(k,x) for k,x in zip(ind, wprobs[ind])]
				#wsorted = sorted(list(enumerate(wprobs)), key=lambda x:x[-1], reverse=True)   # slow
				for wid, wp in wsorted[:topk]: 
					wprob = decoded_logps[prev]+wp
					if wprob < bests.get(ii, -1e5) * early_stop_mult: continue
					cands.append( (prev, wid, wprob) )
			cands.sort(key=lambda x:x[-1], reverse=True)	
			cands = cands[:topk]
			lastks[ii] = len(cands)
			for kk, zz in enumerate(cands):
				prev, wid, wprob = zz
				npos = base+kk
				for k in range(len(next_dec_outputs)):
					next_dec_outputs[k][npos,:,:] = dec_outputs[k][prev]
				target_one[npos,0] = wid
				decoded_logps[npos] = wprob
				next_decoded_indexes[npos] = decoded_indexes[prev].copy()
				next_decoded_indexes[npos].append(wid)
				if wid == end_mark:
					final_results.append( (ii, decoded_indexes[prev].copy(), wprob) ) 
					if ii not in bests or wprob > bests[ii]: bests[ii] = wprob
		if sum(lastks) == 0: break
		dec_outputs = next_dec_outputs
		decoded_indexes = next_decoded_indexes
	return final_results



add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
repeat_last_layer =  Lambda(lambda x: K.concatenate([x, tf.expand_dims(x[-1], axis = -1)]))


def get_hidden_trans_layer(d_model, d_data, num_hidden_trans_layer):
	if num_hidden_trans_layer == 0:
		return(Lambda(lambda x: x, name='xt_reconstruct'))

	else:
		layer_trans = Sequential(name='xt_reconstruct')

		for i in range(num_hidden_trans_layer-1):
			layer_trans.add(TimeDistributed(Dense(d_model, activation='relu')))
			layer_trans.add(TimeDistributed(BatchNormalization()))

		layer_trans.add(TimeDistributed(Dense(d_data)))

		return layer_trans

# class predictor(Layer):
# 	'''
# 	Input z(t) and x(t), calculate the personalized effect mu_t
# 	'''
# 	def __init(self, embed_layer):
# 		self.personalized_encoder = 



class Transformer:
	def __init__(self, len_limit = 50, time_limit = 50, \
			  d_data = 16, d_model=64, d_inner_hid=64, \
			  n_head=4, layers=2, dropout=0.1, \
			  tem_dim = 0, num_states = 3, num_hidden_trans_layer = 2):
		'''
		num_hidden_trans_layer: how many layers to transform ht to xt
		'''

		self.len_limit = len_limit
		self.time_limit = len_limit
		self.d_model = d_model
		self.d_data = d_data
		self.decode_model = None
		self.readout_model = None
		self.layers = layers
		self.tem_dim = tem_dim
		self.num_states = num_states
		d_emb = d_model


		d_k = d_v = d_model // n_head
		assert d_k * n_head == d_model and d_v == d_k

		'''Encoder'''
		with tf.name_scope('encoder'):
			self.input_xt = Input(shape=(self.len_limit, self.d_data+1), dtype='float32', name = 'enc_seq_input')
			self.dense_emb = TimeDistributed(Dense(self.d_model), name = 'encoder_dense_emb')
			self.pos_emb = PosEncodingLayer(time_limit, d_emb)
			self.emb_dropout = Dropout(dropout)
			self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
			self.encoder_dense_zt = TimeDistributed(Dense(num_states), name = 'encoder_dense_zt')

		'''Decoder'''
		with tf.name_scope('decoder'):
			self.embedding_zt = Embedding(num_states, d_model, name='zt_embedding')
			self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
		# self.embedding_sigma_zt  = Embedding(num_states, self.d_sigma, name='zt_embedding_sigma')

		'''Prediction functions'''
		with tf.name_scope('prediction'):
			# self.presonalized_encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
			self.dense_zt = TimeDistributed(Dense(self.num_states), name = 'prediction_zt')
			self.dense_mut = TimeDistributed(Dense(d_model), name = 'prediction_mut') # should I add some other regularization?
			# Simga is a matrix with size (d_model, d_model)
			# self.dense_sigmat = TimeDistributed(Dense(self.d_sigma, name = 'prediction_sigmat'))

		'''Hidden space transform'''
		# self.layer_trans = get_hidden_trans_layer(d_model, d_data, num_hidden_trans_layer)
		with tf.name_scope('transform'):
			self.layer_trans = TimeDistributed(Dense(d_data))
		# self.layer_trans.add(Dense(d_model, activation='relu'))
		# self.layer_trans.add(BatchNormalization())
		# self.layer_trans.add(Dense(d_data))
		# self.layer_trans.add(BatchNormalization())


	def compile(self, optimizer='adam', active_layers=999, loss_weights = [2e-2, 1, 1]):
		'''Encoder'''
		enc_seq_input = self.input_xt # input xt, while the last dim is the time dim

		enc_seq = Lambda(lambda x:x[:,:,1:], name = 'enc_seq')(enc_seq_input)
		self.enc_seq = enc_seq
		self.enc_mask = Lambda(lambda x:tf.cast(tf.reduce_any(K.not_equal(x, 0), axis=-1), 'float32'), name = 'enc_mask')(enc_seq)
		self.enc_time = Lambda(lambda x:x[:,:,0], name = 'enc_time')(enc_seq_input)

		enc_emb = self.dense_emb(enc_seq)
		enc_emb = add_layer([enc_emb, self.pos_emb(self.enc_time, pos_input=True)]) # Here set pos_input=True means input position to calculate positional embeddings
		enc_emb = self.emb_dropout(enc_emb)
		self.enc_emb = enc_emb
		self.enc_output = self.encoder(enc_emb, self.enc_mask, active_layers=active_layers)

		encoder_zt_logits = self.encoder_dense_zt(self.enc_output)
		zt = Softmax()(encoder_zt_logits)
		self.zt = zt

		self.model = Model(enc_seq_input, zt)

		'''sampling'''
		encoder_sampled_zt = Lambda(lambda x: tfp.distributions.Sample(tfp.distributions.Categorical(x)).sample(), name = 'encoder_sampled_zt')(zt)
		self.encoder_sampled_zt = encoder_sampled_zt

		'''Decoder'''
		self.dec_seq_input = encoder_sampled_zt
		self.dec_seq  = Lambda(lambda x:x[:,:], name='self.dec_seq')(self.dec_seq_input)
		self.delay_mask = Lambda(lambda x: K.concatenate([x[:,1:], tf.expand_dims(x[:,-1], axis=-1)]), name='dec_delay_mask')(self.enc_mask) # repeat the last element
		dec_delay_zt = Lambda(lambda x: K.concatenate([x[:,1:], tf.expand_dims(x[:,-1], axis=-1)]), name='dec_delay_zt')(self.dec_seq_input) # repeat the last element
		dec_delay_time = Lambda(lambda x: K.concatenate([x[:,1:], tf.expand_dims(x[:,-1], axis=-1)]), name='dec_delay_time')(self.enc_time)
		self.dec_delay_zt = dec_delay_zt

		self.mu_zt = self.embedding_zt(self.dec_seq)
		# self.sigma_zt = self.embedding_sigma_zt(self.dec_seq)
		
		dec_emb = add_layer([self.mu_zt, self.pos_emb(dec_delay_time, pos_input=True)]) # Here set pos_input=True means input position to calculate positional embeddings
		dec_emb = self.mu_zt

		# Here use self.enc_mask to generate mask
		dec_output = self.decoder(dec_emb, self.enc_mask, self.enc_mask, self.enc_output, active_layers=active_layers) 
		self.dec_output = dec_output

		'''Prediction functions'''
		# self.predict_delay_zt_logits = add_layer([dec_output, self.pos_emb(dec_delay_time, pos_input=True)])
		self.predict_delay_zt_logits = dec_output # Here I do not count temporal information
		self.predict_delay_zt_logits = self.dense_zt(self.predict_delay_zt_logits)
		self.predict_mut = self.dense_mut(dec_output)
		# self.predict_mut = dec_output
		# self.predict_sigmat = self.dense_sigmat(dec_output)

		self.mu_t = Add()([self.mu_zt, self.predict_mut])
		# self.sigma_t = Add()([self.sigma_zt, self.predict_sigmat])
		# self.sigma_t = Reshape([self.len_limit, self.d_model, self.d_model])(self.sigma_t)
		# self.sigma_t = Lambda(lambda x: tf.linalg.band_part(x,-1,1))(self.sigma_t) 

		'''Hidden space transform'''
		self.x_rec_t = self.layer_trans(self.mu_t)



		'''ELBO'''
		mask = self.enc_mask
		delay_mask = self.delay_mask

		'''
		p(x(t)|z(t),x(t-1),z(t-1))
		Use MSE to replace it
		'''
		def get_ELBO_part1():
			# GM = tfp.distributions.MultivariateNormalTriL(
			# 	loc = self.mu_t, 
			# 	scale_tril= self.sigma_t)
			# p_h = GM.prob(self.ht)
			# loss = K.log(p_h) #+ K.log(self.d_jacobian)
			# loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			# loss = Lambda(lambda x: K.mean(x))(loss)
			print('Use MSE to replace the emission function')
			loss = mean_squared_error(enc_seq, self.x_rec_t)
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss # return the negative MSE

		'''p(z(t)|z(t-1),x(t-1))'''
		def get_ELBO_part2(q_version = False):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = dec_delay_zt, logits = self.predict_delay_zt_logits )
			loss = tf.reduce_sum(loss * delay_mask, -1) / tf.reduce_sum(delay_mask, -1)
			loss = K.mean(loss)
			return loss # cross entropy has a negative sign


		'''p(z(t)|x(t))'''
		def get_ELBO_part3(q_version = False):
			if q_version is False:
				loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = encoder_sampled_zt, logits = encoder_zt_logits)
				loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
				loss = K.mean(loss)
				return loss # cross entropy has a negative sign
			else:
				loss = mean_squared_error(self.enc_output, self.mu_zt)
				loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
				loss = K.mean(loss)

		self.loss1 = get_ELBO_part1()
		self.loss2 = get_ELBO_part2()
		self.loss3 = get_ELBO_part3()

		self.ELBO = loss_weights[0] * self.loss1 + loss_weights[1] * self.loss2 + loss_weights[2] * self.loss3

		def compute_decoder_predict_zt_acc():
			corr = K.cast(K.equal(K.cast(dec_delay_zt, 'int32'), K.cast(K.argmax(self.predict_delay_zt_logits, axis=-1), 'int32')), 'float32')
			corr = K.sum(corr * delay_mask, -1) / K.sum(delay_mask, -1)
			return K.mean(corr)

		self.decoder_predict_zt_acc = compute_decoder_predict_zt_acc()

		'''Compile model'''
		self.model = Model(self.input_xt, zt)
		self.model.add_loss(self.ELBO)

		self.model.add_metric(self.decoder_predict_zt_acc, 'Decoder pre acc')
		self.model.add_metric(self.loss1, 'loss1')
		self.model.add_metric(self.loss2, 'loss2')
		self.model.add_metric(self.loss3, 'loss3')

		self.model.compile(optimizer= RAdam(lr=0.001))

	
	def justify_model(self, lr = 0.001, loss_weights = [2e-2,1,1]):
		'''
		We recommend to only just the first loss term
		'''
		self.model = Model(self.input_xt, self.zt)
		self.ELBO = loss_weights[0] * self.loss1 + loss_weights[1] * self.loss2 + loss_weights[2] * self.loss3
		self.model.add_loss([self.ELBO])

		self.model.add_metric(self.decoder_predict_zt_acc, 'Decoder pre acc')
		self.model.add_metric(self.loss1, 'loss1')
		self.model.add_metric(self.loss2, 'loss2')
		self.model.add_metric(self.loss3, 'loss3')

		self.model.compile(optimizer= RAdam(lr = lr))
	
	def init_model(self, X, active_layers = 999, batch_size = 128):
		'''
		First, train model with only the reconstuction loss (loss 1), building the hidden space.
		Then, train a KMeans or GM on the hidden space to get states.
		Finally, train the encoder to predict the trained states with.
		'''
		self.init_model_stage_one(X)
		self.init_model_stage_two(X)
	
	def init_model_stage_one(self, X, active_layers = 999, batch_size = 128):
		'''Step one'''
		print('Start training encoder'+'-'*30)
		init_input = self.enc_output
		# init_mut = self.dense_mut(init_input)
		init_recon_xt = self.layer_trans(init_input) # Note here recon_xt should introduce temporal information

		mask = self.enc_mask
		loss = mean_squared_error(self.enc_seq, init_recon_xt)
		loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
		loss = K.mean(loss)

		self.recon_model = Model(self.input_xt, init_recon_xt)
		self.recon_model.add_loss(loss)
		self.recon_model.compile(RAdam(1e-3))
		self.recon_model.fit(X, batch_size = batch_size, epochs = 50)
		self.recon_model.compile(RAdam(1e-4))
		self.recon_model.fit(X, batch_size = batch_size, epochs = 50)
		print('Finish training reconstruction model'+'-'*30)


	def init_model_stage_two(self, X, batch_size = 128):
		'''Step two'''

		print('Start getting hidden space'+'-'*30)
		self.hidden_model = Model(self.input_xt, self.enc_output)
		X_hidden = self.hidden_model.predict(X)

		X_hidden_ = []
		lens = []
		for index, item in enumerate(X):
			l = int(max(item[:,0]))+1
			lens.append(l)
			for i in range(l):
				X_hidden_.append(X_hidden[index,i])

		print('Finish getting hidden space'+'-'*30)
						

		print('Start KMeans init'+'-'*30)
		from sklearn.cluster import KMeans
		self.kmeans = KMeans(self.num_states, n_jobs = 4, n_init = 3).fit(X_hidden_)

		initial_mu = self.kmeans.cluster_centers_
		self.embedding_zt.set_weights([initial_mu])

		initial_states_ = self.kmeans.predict(X_hidden_)
		initial_states = []
		current_l = 0
		for index, l in enumerate(lens):
			s = initial_states_[current_l:current_l+l]
			current_l = current_l+l
			s = list(s) + [s[-1]] * (self.len_limit - l)
			initial_states.append(s)

		self.initial_states = np.array(initial_states).reshape(-1, self.len_limit)

		print('Finish KMeans init'+'-'*30)
		

		'''Step three'''
		print('Start training encoder'+'-'*30)

		state_true = Input(shape=(self.len_limit,self.num_states),dtype='int32')
			
		self.classify_model = Model([self.input_xt, state_true], self.zt)

		loss = categorical_crossentropy(state_true, self.zt)
		loss = tf.reduce_sum(loss * self.enc_mask, -1) / tf.reduce_sum(self.enc_mask, -1)
		loss = K.mean(loss)

		self.classify_model.add_loss(loss)

		self.classify_model.compile(optimizer = 'adam', metrics = ['acc'])
		self.classify_model.fit([X, np_utils.to_categorical(self.initial_states)], batch_size=batch_size, epochs=50)
		print('Finish training encoder'+'-'*30)


		print('Finish initialization'+'-'*30)

		
	def init_model_stage_three(self, X, batch_size = 128):
		'''stage one + stage two'''

		self.stage_three_model = Model(self.input_xt,self.layer_trans(self.mu_zt))

		# Reconstruction loss
		def get_mse_loss():
			mask = self.enc_mask
			loss = mean_squared_error(self.enc_seq, self.stage_three_model.output)
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss

		# quantization loss
		def get_quantization_loss():
			mask = self.enc_mask
			loss = mean_squared_error(self.enc_output, self.mu_zt)
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss

		loss1 = get_mse_loss()
		# loss2 = get_quantization_loss()

		self.stage_three_model.add_loss(loss1)
		# self.stage_three_model.add_loss(loss2)
		self.stage_three_model.add_metric(loss1,'mse')
		# self.stage_three_model.add_metric(loss2,'quantization')
		self.stage_three_model.compile(RAdam(1e-3))

		self.stage_three_model.fit(X, batch_size = batch_size, epochs = 50)

	

class PosEncodingLayer:
	def __init__(self, max_len, d_emb):
		self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
						   weights=[GetPosEncodingMatrix(max_len, d_emb)])
	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask
	def __call__(self, seq, pos_input=False):
		x = seq
		if not pos_input: x = Lambda(self.get_pos_seq)(x)
		return self.pos_emb_matrix(x)


class AddPosEncoding:
	def __call__(self, x):
		_, max_len, d_emb = K.int_shape(x)
		pos = GetPosEncodingMatrix(max_len, d_emb)
		x = Lambda(lambda x:x+pos)(x)
		return x

class LRSchedulerPerStep(Callback):
	def __init__(self, d_model, warmup=4000):
		self.basic = d_model**-0.5
		self.warm = warmup**-1.5
		self.step_num = 0
	def on_batch_begin(self, batch, logs = None):
		self.step_num += 1
		lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
		K.set_value(self.model.optimizer.lr, lr)


if __name__ == '__main__':
	print('done')