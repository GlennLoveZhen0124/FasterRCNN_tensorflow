import tensorflow as tf
import numpy as np
import os


class Backbone(object):
	def __init__(self, train_process_step):
		self.train_process_step = train_process_step
		assert train_process_step in [1,2,3,4, 'inference2', 'inference4']
		if self.train_process_step == 1:
			self.trainable_list = [False, False, False, False, True, True, True, True, True, True, True, True, True]
		elif self.train_process_step == 2:
			self.trainable_list = [False, False, False, False, True, True, True, True, True, True, True, True, True]
		elif self.train_process_step == 3:
			self.trainable_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
		elif self.train_process_step == 4:
			self.trainable_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
		elif self.train_process_step == 'inference2':
			self.trainable_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
		elif self.train_process_step == 'inference4':
			self.trainable_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
		
		print('Initializing VGG16 model ... ...')
		
	def weights_init(self, shape, trainable, name):
		var = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def biases_init(self, shape, trainable, name):
		var = tf.constant(0.1, shape=shape)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def build(self, input_tensor):
		with tf.variable_scope('vgg_16'):
			with tf.variable_scope('conv1'):
				with tf.variable_scope('conv1_1'):
					weights1_1 = self.weights_init([3,3,3,64], self.trainable_list[0], 'weights')
					biases1_1 = self.biases_init([64], self.trainable_list[0], 'biases')
				net = tf.nn.conv2d(input_tensor, weights1_1, strides=[1,1,1,1], padding='SAME') + biases1_1
				net = tf.nn.relu(net)
				with tf.variable_scope('conv1_2'):
					weights1_2 = self.weights_init([3,3,64,64], self.trainable_list[1], 'weights')
					biases1_2 = self.biases_init([64], self.trainable_list[1], 'biases')
				net = tf.nn.conv2d(net, weights1_2, strides=[1,1,1,1], padding='SAME') + biases1_2
				net = tf.nn.relu(net)
			net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
			with tf.variable_scope('conv2'):
				with tf.variable_scope('conv2_1'):
					weights2_1 = self.weights_init([3,3,64,128], self.trainable_list[2], 'weights')
					biases2_1 = self.biases_init([128], self.trainable_list[2], 'biases')
				net = tf.nn.conv2d(net, weights2_1, strides=[1,1,1,1], padding='SAME') + biases2_1
				net = tf.nn.relu(net)
				with tf.variable_scope('conv2_2'):
					weights2_2 = self.weights_init([3,3,128,128], self.trainable_list[3], 'weights')
					biases2_2 = self.biases_init([128], self.trainable_list[3], 'biases')
				net = tf.nn.conv2d(net, weights2_2, strides=[1,1,1,1], padding='SAME') + biases2_2
				net = tf.nn.relu(net)
			net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
			with tf.variable_scope('conv3'):
				with tf.variable_scope('conv3_1'):
					weights3_1 = self.weights_init([3,3,128,256], self.trainable_list[4], 'weights')
					biases3_1 = self.biases_init([256], self.trainable_list[4], 'biases')
				net = tf.nn.conv2d(net, weights3_1, strides=[1,1,1,1], padding='SAME') + biases3_1
				net = tf.nn.relu(net)
				with tf.variable_scope('conv3_2'):
					weights3_2 = self.weights_init([3,3,256,256], self.trainable_list[5], 'weights')
					biases3_2 = self.biases_init([256], self.trainable_list[5], 'biases')
				net = tf.nn.conv2d(net, weights3_2, strides=[1,1,1,1], padding='SAME') + biases3_2
				net = tf.nn.relu(net)
				with tf.variable_scope('conv3_3'):
					weights3_3 = self.weights_init([3,3,256,256], self.trainable_list[6], 'weights')
					biases3_3 = self.biases_init([256], self.trainable_list[6], 'biases')
				net = tf.nn.conv2d(net, weights3_3, strides=[1,1,1,1], padding='SAME') + biases3_3
				net = tf.nn.relu(net)
			net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
			with tf.variable_scope('conv4'):
				with tf.variable_scope('conv4_1'):
					weights4_1 = self.weights_init([3,3,256,512], self.trainable_list[7], 'weights')
					biases4_1 = self.biases_init([512], self.trainable_list[7], 'biases')
				net = tf.nn.conv2d(net, weights4_1, strides=[1,1,1,1], padding='SAME') + biases4_1
				net = tf.nn.relu(net)
				with tf.variable_scope('conv4_2'):
					weights4_2 = self.weights_init([3,3,512,512], self.trainable_list[8], 'weights')
					biases4_2 = self.biases_init([512], self.trainable_list[8], 'biases')
				net = tf.nn.conv2d(net, weights4_2, strides=[1,1,1,1], padding='SAME') + biases4_2
				net = tf.nn.relu(net)
				with tf.variable_scope('conv4_3'):
					weights4_3 = self.weights_init([3,3,512,512], self.trainable_list[9], 'weights')
					biases4_3 = self.biases_init([512], self.trainable_list[9], 'biases')
				net = tf.nn.conv2d(net, weights4_3, strides=[1,1,1,1], padding='SAME') + biases4_3
				net = tf.nn.relu(net)
			net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
			with tf.variable_scope('conv5'):
				with tf.variable_scope('conv5_1'):
					weights5_1 = self.weights_init([3,3,512,512], self.trainable_list[10], 'weights')
					biases5_1 = self.biases_init([512], self.trainable_list[10], 'biases')
				net = tf.nn.conv2d(net, weights5_1, strides=[1,1,1,1], padding='SAME') + biases5_1
				net = tf.nn.relu(net)
				with tf.variable_scope('conv5_2'):
					weights5_2 = self.weights_init([3,3,512,512], self.trainable_list[11], 'weights')
					biases5_2 = self.biases_init([512], self.trainable_list[11], 'biases')
				net = tf.nn.conv2d(net, weights5_2, strides=[1,1,1,1], padding='SAME') + biases5_2
				net = tf.nn.relu(net)
				with tf.variable_scope('conv5_3'):
					weights5_3 = self.weights_init([3,3,512,512], self.trainable_list[12], 'weights')
					biases5_3 = self.biases_init([512], self.trainable_list[12], 'biases')
				net = tf.nn.conv2d(net, weights5_3, strides=[1,1,1,1], padding='SAME') + biases5_3
				net = tf.nn.relu(net)
		return net


if __name__ == '__main__':
	arr = np.ones((32,224,224,3))
	arr = arr.astype(np.float32)
	input_tensor = tf.Variable(arr, tf.float32)
	BB = Backbone(train_process_step = 1)
	feature_map = BB.build(input_tensor)
	print('feature map: ', feature_map)




