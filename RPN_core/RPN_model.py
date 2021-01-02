import tensorflow as tf
import numpy as np
import os


class VGGModel(object):
	def __init__(self, train_process_step):
		self.train_process_step = train_process_step
		assert train_process_step in [1,3]
		if self.train_process_step == 1:
			self.trainable_list = [False, False, False, False, True, True, True, True, True, True, True, True, True]
		elif self.train_process_step == 3:
			self.trainable_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
		print('Initializing VGG16 model ... ...')
		
	def weights_init(self, shape, trainable, name):
		var = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def biases_init(self, shape, trainable, name):
		var = tf.constant(0.1, shape=shape)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def vgg16(self, input_tensor):
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
	
	def RPN_head(self, x, RPN_k = 9):
		with tf.variable_scope('RPN_network'):
			net = tf.layers.conv2d(x, filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu, name = 'RPN_conv3x3')
			net_cls = tf.layers.conv2d(net, filters = 2*RPN_k, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, name = 'RPN_cls_conv1x1')
			net_cls_shape = net_cls.get_shape().as_list()
			net_cls = tf.reshape(net_cls, (net_cls_shape[0],net_cls_shape[1],net_cls_shape[2]*RPN_k,net_cls_shape[3]//RPN_k))
			net_cls = tf.nn.softmax(net_cls)
			net_cls = tf.reshape(net_cls, net_cls_shape)
			net_reg = tf.layers.conv2d(net, filters = 4*RPN_k, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, name = 'RPN_reg_conv1x1')
		return net_cls, net_reg	# net_cls: (1, 37, 50, 24), net_reg: (1, 37, 50, 48)
	
	def decode_feature_map(self, net_cls, net_reg, RPN_k = 9):	# RPN_k = 12
		net_cls_shape = net_cls.get_shape().as_list()
		net_reg_shape = net_reg.get_shape().as_list()
		net_reg_x = tf.strided_slice(net_reg, [0,0,0,0], [net_reg_shape[0],net_reg_shape[1], net_reg_shape[2], net_reg_shape[3]], [1,1,1,4])
		print('net_reg_x: ', net_reg_x.shape)
		net_reg_y = tf.strided_slice(net_reg, [0,0,0,1], [net_reg_shape[0],net_reg_shape[1], net_reg_shape[2], net_reg_shape[3]], [1,1,1,4])
		print('net_reg_y: ', net_reg_y.shape)
		net_reg_w = tf.strided_slice(net_reg, [0,0,0,2], [net_reg_shape[0],net_reg_shape[1], net_reg_shape[2], net_reg_shape[3]], [1,1,1,4])
		print('net_reg_w: ', net_reg_w.shape)
		net_reg_h = tf.strided_slice(net_reg, [0,0,0,3], [net_reg_shape[0],net_reg_shape[1], net_reg_shape[2], net_reg_shape[3]], [1,1,1,4])
		print('net_reg_h: ', net_reg_h.shape)
		net_cls_0 = tf.strided_slice(net_cls, [0,0,0,0], [net_cls_shape[0], net_cls_shape[1], net_cls_shape[2], net_cls_shape[3]], [1,1,1,2])
		net_cls_1 = tf.strided_slice(net_cls, [0,0,0,1], [net_cls_shape[0], net_cls_shape[1], net_cls_shape[2], net_cls_shape[3]], [1,1,1,2])
		net_cls_ret = tf.stack([net_cls_0, net_cls_1], axis=-1)
		print('net_cls: ', net_cls.shape)
		print('---net_cls_0---: ', net_cls_0.shape)
		print('---net_cls_1---: ', net_cls_1.shape)
		print('---net_cls_ret---: ', net_cls_ret.shape)
		#return net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_ret
		return net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1



if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	'''
	VM = VGGModel()
	x = np.ones([128,224,224,3], dtype=np.float32)
	print(x,x.shape,x.dtype)
	model = VM.vgg16(x)
	print(model)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.save(sess, os.path.join('./test_model','test_model.ckpt'))
	'''
	
	x = np.ones([1,640,640,3], dtype=np.float32)
	pretrain_checkpoint = '/root/Faster_RCNN_tensorflow_iterative_train/VGG/vgg_16.ckpt'
	VM = VGGModel()
	vgg_logit = VM.vgg16(x)
	net_cls, net_reg = VM.RPN_head(vgg_logit, RPN_k=12)
	global_vars = tf.global_variables()
	print('global_vars: ', global_vars)
	var_list = [v for v in global_vars if 'vgg_16' in v.name and 'mean_rgb' not in v.name and 'fc' not in v.name]
	saver = tf.train.Saver(var_list = var_list)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, pretrain_checkpoint)







