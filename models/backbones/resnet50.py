import tensorflow as tf
import numpy as np


class Backbone(object):
	def __init__(self, train_process_step):
		assert train_process_step in [1,2,3,4, 'inference2', 'inference4']
		self.train_process_step = train_process_step
		if self.train_process_step == 1:			# load resnet50 premodel, train loc and shp branches
			self.train_status_backbone = dict(conv1 = True, block1 = True, block2 = True, block3 = True, block4 = True)
			self.trainable_dict_backbone = dict(conv1 = True, block1 = True, block2 = True, block3 = True, block4 = True)
		elif self.train_process_step == 2:
			self.train_status_backbone = dict(conv1 = True, block1 = True, block2 = True, block3 = True, block4 = True)
			self.trainable_dict_backbone = dict(conv1 = True, block1 = True, block2 = True, block3 = True, block4 = True)
		elif self.train_process_step == 3:
			self.train_status_backbone = dict(conv1 = True, block1 = True, block2 = True, block3 = True, block4 = True)
			self.trainable_dict_backbone = dict(conv1 = False, block1 = False, block2 = False, block3 = False, block4 = False)
		elif self.train_process_step == 4:
			self.train_status_backbone = dict(conv1 = True, block1 = True, block2 = True, block3 = True, block4 = True)
			self.trainable_dict_backbone = dict(conv1 = False, block1 = False, block2 = False, block3 = False, block4 = False)
		elif self.train_process_step == 'inference2':
			self.train_status_backbone = dict(conv1 = False, block1 = False, block2 = False, block3 = False, block4 = False)
			self.trainable_dict_backbone = dict(conv1 = False, block1 = False, block2 = False, block3 = False, block4 = False)
		elif self.train_process_step == 'inference4':
			self.train_status_backbone = dict(conv1 = False, block1 = False, block2 = False, block3 = False, block4 = False)
			self.trainable_dict_backbone = dict(conv1 = False, block1 = False, block2 = False, block3 = False, block4 = False)
	
	def weights_init_backbone(self, shape, trainable, name):
		var = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def biases_init_backbone(self, shape, trainable, name):
		var = tf.constant(0.1, shape=shape)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def bottleneck(self, x, input_dim, f, filters, trainable = True, train_status_backbone = True, s = None, stage = None, is_identity_block = False):
		F1, F2, F3 = filters
		net_shortcut = x
		with tf.variable_scope('conv1'):
			weights1 = self.weights_init_backbone(shape=[1,1,input_dim,F1], trainable=trainable, name = 'weights')
			if is_identity_block:
				net = tf.nn.conv2d(x, weights1, strides=[1,1,1,1], padding='VALID')
			else:
				net = tf.nn.conv2d(x, weights1, strides=[1,s,s,1], padding='VALID')
			net = tf.layers.batch_normalization(net, training = train_status_backbone, name = 'BatchNorm')
			net = tf.nn.relu(net)
		with tf.variable_scope('conv2'):
			weights2 = self.weights_init_backbone(shape=[f,f,F1,F2], trainable=trainable, name = 'weights')
			net = tf.nn.conv2d(net, weights2, strides=[1,1,1,1], padding='SAME')
			net = tf.layers.batch_normalization(net, training = train_status_backbone, name = 'BatchNorm')
			net = tf.nn.relu(net)
		with tf.variable_scope('conv3'):
			weights3 = self.weights_init_backbone(shape=[1,1,F2,F3], trainable=trainable, name = 'weights')
			net = tf.nn.conv2d(net, weights3, strides=[1,1,1,1], padding='VALID')
			net = tf.layers.batch_normalization(net, training = train_status_backbone, name = 'BatchNorm')
		if not is_identity_block:
			with tf.variable_scope('shortcut'):
				weights4 = self.weights_init_backbone(shape=[1,1,input_dim,F3], trainable=trainable, name = 'weights')
				net_shortcut = tf.nn.conv2d(net_shortcut, weights4, strides=[1,s,s,1], padding='VALID')
				net_shortcut = tf.layers.batch_normalization(net_shortcut, training = train_status_backbone, name = 'BatchNorm')
		net = net + net_shortcut
		net = tf.nn.relu(net)
		return net
	
	def build_backbone(self, input_tensor, input_dim = 3):
		print('self.trainable_dict_backbone: ', self.trainable_dict_backbone)
		net = tf.pad(input_tensor, [[0,0], [4,4], [4,4], [0,0]])
		#net = input_tensor
		with tf.variable_scope('resnet_v1_50'):
			with tf.variable_scope('conv1'):
				weights_ini = self.weights_init_backbone(shape=[7,7,input_dim,64], trainable=self.trainable_dict_backbone['conv1'], name = 'weights')
				net = tf.nn.conv2d(net, weights_ini, strides=[1,2,2,1], padding='VALID')
				net = tf.layers.batch_normalization(net, training = self.train_status_backbone['conv1'], name = 'BatchNorm')
				net = tf.nn.relu(net)
				net = tf.nn.max_pool(net, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
			with tf.variable_scope('block1'):
				with tf.variable_scope('unit_1'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=64, f=3, filters=[64,64,256], trainable=self.trainable_dict_backbone['block1'], train_status_backbone = self.train_status_backbone['block1'], s = 1, is_identity_block = False)	# unit_1
				with tf.variable_scope('unit_2'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=256, f=3, filters=[64,64,256], trainable=self.trainable_dict_backbone['block1'], train_status_backbone = self.train_status_backbone['block1'], is_identity_block = True)		# unit_2
				with tf.variable_scope('unit_3'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=256, f=3, filters=[64,64,256], trainable=self.trainable_dict_backbone['block1'], train_status_backbone = self.train_status_backbone['block1'], is_identity_block = True)		# unit_3
			net1 = net
			with tf.variable_scope('block2'):
				with tf.variable_scope('unit_1'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=256, f=3, filters=[128,128,512], trainable=self.trainable_dict_backbone['block2'], train_status_backbone = self.train_status_backbone['block2'], s = 2, is_identity_block = False)	# unit_1
				with tf.variable_scope('unit_2'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=512, f=3, filters=[128,128,512], trainable=self.trainable_dict_backbone['block2'], train_status_backbone = self.train_status_backbone['block2'], is_identity_block = True)		# unit_2
				with tf.variable_scope('unit_3'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=512, f=3, filters=[128,128,512], trainable=self.trainable_dict_backbone['block2'], train_status_backbone = self.train_status_backbone['block2'], is_identity_block = True)		# unit_3
				with tf.variable_scope('unit_4'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=512, f=3, filters=[128,128,512], trainable=self.trainable_dict_backbone['block2'], train_status_backbone = self.train_status_backbone['block2'], is_identity_block = True)		# unit_4
			net2 = net
			with tf.variable_scope('block3'):
				with tf.variable_scope('unit_1'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=512, f=3, filters=[256,256,1024], trainable=self.trainable_dict_backbone['block3'], train_status_backbone = self.train_status_backbone['block3'], s = 2, is_identity_block = False)	# unit_1
				with tf.variable_scope('unit_2'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=1024, f=3, filters=[256,256,1024], trainable=self.trainable_dict_backbone['block3'], train_status_backbone = self.train_status_backbone['block3'], is_identity_block = True)		# unit_2
				with tf.variable_scope('unit_3'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=1024, f=3, filters=[256,256,1024], trainable=self.trainable_dict_backbone['block3'], train_status_backbone = self.train_status_backbone['block3'], is_identity_block = True)		# unit_3
				with tf.variable_scope('unit_4'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=1024, f=3, filters=[256,256,1024], trainable=self.trainable_dict_backbone['block3'], train_status_backbone = self.train_status_backbone['block3'], is_identity_block = True)		# unit_4
				with tf.variable_scope('unit_5'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=1024, f=3, filters=[256,256,1024], trainable=self.trainable_dict_backbone['block3'], train_status_backbone = self.train_status_backbone['block3'], is_identity_block = True)		# unit_5
				with tf.variable_scope('unit_6'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=1024, f=3, filters=[256,256,1024], trainable=self.trainable_dict_backbone['block3'], train_status_backbone = self.train_status_backbone['block3'], is_identity_block = True)		# unit_6
			net3 = net
			with tf.variable_scope('block4'):
				with tf.variable_scope('unit_1'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=1024, f=3, filters=[512,512,2048], trainable=self.trainable_dict_backbone['block4'], train_status_backbone = self.train_status_backbone['block4'], s = 2, is_identity_block = False)	# unit_1
				with tf.variable_scope('unit_2'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=2048, f=3, filters=[512,512,2048], trainable=self.trainable_dict_backbone['block4'], train_status_backbone = self.train_status_backbone['block4'], is_identity_block = True)		# unit_2
				with tf.variable_scope('unit_3'):
					with tf.variable_scope('bottleneck_v1'):
						net = self.bottleneck(net, input_dim=2048, f=3, filters=[512,512,2048], trainable=self.trainable_dict_backbone['block4'], train_status_backbone = self.train_status_backbone['block4'], is_identity_block = True)		# unit_3
			net4 = net
		
		return net1, net2, net3, net4


if __name__ == '__main__':
	arr = np.ones((32,224,224,3))
	arr = arr.astype(np.float32)
	input_tensor = tf.Variable(arr, tf.float32)
	BB = Backbone(train_process_step = 1)
	feature_map = BB.build_backbone(input_tensor)
	print('feature map: ', feature_map)




