import tf_util as U
import tensorflow as tf
import numpy as np

# Check max-pooling on proj, deconv net


def proj_net(scope, img, latent_dim):
	with tf.variable_scope(scope):
		x = img
		x = tf.nn.relu(U.conv2d(x, 64, "l1", [8, 8], [2, 2], pad="VALID"))
		x = tf.nn.relu(U.conv2d(x, 128, "l2", [6, 6], [2, 2], pad="VALID"))
		x = tf.nn.relu(U.conv2d(x, 128, "l3", [6, 6], [2, 2], pad="VALID"))
		x = tf.nn.relu(U.conv2d(x, 128, "l4", [4, 4], [2, 2], pad="VALID"))
		x = U.flattenallbut0(x)
		x = tf.nn.relu(U.dense(x, 2048, 'lin1', U.normc_initializer(1.0)))
		x = U.dense(x, latent_dim, 'lin2', U.normc_initializer(1.0))
		return x

def deconv_net(scope, latent_variable):
	with tf.variable_scope(scope):
		x = latent_variable
		x = U.dense(x, 2048, 'lin3', U.normc_initializer(1.0))
		x = tf.nn.relu(U.dense(x, 2048, 'lin4', U.normc_initializer(1.0)))
		x = tf.reshape(x, []) # Unflatten
		x = tf.nn.relu(U.conv2d(x, 128, "l5", [4, 4], [2, 2], pad="VALID"))
		x = tf.nn.relu(U.conv2d(x, 128, "l6", [6, 6], [2, 2], pad="VALID"))
		x = tf.nn.relu(U.conv2d(x, 128, "l7", [6, 6], [2, 2], pad="VALID"))
		x = U.conv2d(x, 3, "l8", [8, 8], [2, 2], pad="VALID")
		return x

class mymodel(object):
	def __init__(self, name, *args, **kwargs):
		with tf.variable_scope(name):
			self._init(*args, **kwargs)
			self.scope = tf.get_variable_scope().name

	def _init(self, img_shape, latent_dim):
		sequence_length = None
		img1 = U.get_placeholder(name="img1", dtype=tf.float32, shape=[sequence_length] + img_shape)
		img2 = U.get_placeholder(name="img2", dtype=tf.float32, shape=[sequence_length] + img_shape)

		img1_scaled = img1
		img2_scaled = img2

		[latent1, latent2] = feat_matching(img1_scaled, img2_scaled, latent_dim)
		self.match_error = U.mean(tf.square(latent1 - latent2))
		[reconst1, reconst2] = reconstruct(latent1, latent2)

		self.reconst_error1 = U.mean(tf.square(reconst1 - img1_scaled))
		self.reconst_error2 = U.mean(tf.square(reconst2 - img2_scaled))

		self._get_loss = U.function([img1, img2], [self.match_error, self.reconst_error1, self.reconst_error2])
		self._get_reconst_img = U.function([img1, img2], [reconst1, reconst2])

	def feat_matching(self, s1, s2, latent_dim):
		l1 = proj_net(scope = "proj1", s1, latent_dim)
		l2 = proj_net(scope = "proj2", s2, latent_dim)
		return [l1, l2]

	def get_latent_error(self):
		return self.match_error

	def reconstruct(self, l1, l2):
		reconst1 = deconv_net(scope = "unproj1", l1)
		reconst2 = deconv_net(scope = "unproj2", l2)
		return [reconst1, reconst2]

	def get_reconst_img(self, img1, img2):
		return self._get_reconst_img(img1, img2)

	def get_reconstruct_error(self):
		return [self.reconst_error1, self.reconst_error2]

	def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

	
