import tensorflow as tf
import numpy as np
import tf_util as U

sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 19, 25, 128]
strides = [1, 2, 2, 1]

w = tf.constant(0.1, shape=[4,4,128,128])

output = tf.constant(0.1, shape=output_shape)
expected_l = tf.nn.conv2d(output, w, strides=strides, padding = "VALID")
print expected_l.get_shape()