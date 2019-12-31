import keras.backend as K
from keras.layers import Layer
# from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow as tf

class pdf_transform_layer(Layer):
    def call(self, inputs):
        ht, inc_seq = inputs
        ht_grad = jacobian(ht, inc_seq)
        ht_grad = ht_grad[0]
        self.shape = ht_grad.shape
        return ht_grad
    
    def compute_output_shape(self, input_shape):
        # print(input_shape[1])
        return input_shape[1,:-1]
