import tf_util as U
import tensorflow as tf
import numpy as np
import net

def proj_net(scope, img):

def reconst_net(scope, latent_variable):


class model(object):
	def __init__(self, name, *args, **kwargs):
		with tf.variable_scope(name):
			self._init(*args, **kwargs)
			self.scope = tf.get_variable_scope().name

	def _init(self, img_shape, latent_dim):
		sequence_length = None
		input1 = U.get_placeholder(name="input1", dtype=tf.float32, shape=[sequence_length] + img_shape)
		input2 = U.get_placeholder(name="input2", dtype=tf.float32, shape=[sequence_length] + img_shape)

		input1_scaled = input1
		input2_scaled = input2

		[latent1, latent2] = feat_matching(input1_scaled, input2_scaled, latent_dim)
		self.match_error = U.mean(tf.square(latent1 - latent2))
		[reconst1, reconst2] = reconstruct(latent1, latent2)
		self.reconst_error1 = U.mean(tf.square(reconst1 - input1_scaled))
		self.reconst_error2 = U.mean(tf.square(reconst2 - input2_scaled))

	def feat_matching(self, s1, s2, latent_dim):
		with tf.variable_scope(img1):
			l1 = proj_net(s1)
		with tf.variable_scope(img2):
			l2 = proj_net(s2)
		return [l1, l2]

	def get_latent_error(self):
		return self.match_error


	def reconstruct(self, l1, l2):
		with tf.variable_scope(l1):
			reconst1 = reconst_net(l1)
		with tf.variable_scope(l2):
			reconst2 = proj_net(l2)
		return [reconst1, reconst2]

	def get_reconstruct_error(self):
		return [self.reconst_error1, self.reconst_error2]



	
