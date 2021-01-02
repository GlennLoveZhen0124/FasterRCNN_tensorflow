import tensorflow as tf
import numpy as np


class FasterRCNNModel(object):
	def __init__(self, train_step):
		assert train_step in [2,4]
		self.train_step = train_step
		if self.train_step == 2:
			self.trainable_list = [False, False, False, False, True, True, True, True, True, True, True, True, True]		# Total : 13, Freeze : 4
		elif self.train_step == 4:
			self.trainable_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
		print('Done creating FastRCNN model instance.')
	
	def IOU(self, pred_y1, pred_x1, pred_y2, pred_x2, GT_y1, GT_x1, GT_y2, GT_x2, nms_num):
		GT_y1 = tf.reshape(GT_y1, [-1,1])
		GT_y1 = tf.tile(GT_y1, [1, nms_num])
		GT_x1 = tf.reshape(GT_x1, [-1,1])
		GT_x1 = tf.tile(GT_x1, [1, nms_num])
		GT_y2 = tf.reshape(GT_y2, [-1,1])
		GT_y2 = tf.tile(GT_y2, [1, nms_num])
		GT_x2 = tf.reshape(GT_x2, [-1,1])
		GT_x2 = tf.tile(GT_x2, [1, nms_num])
		
		area1 = (pred_y2 - pred_y1) * (pred_x2 - pred_x1)
		area2 = (GT_y2 - GT_y1) * (GT_x2 - GT_x1)
		left = tf.maximum(pred_x1, GT_x1)
		right = tf.minimum(pred_x2, GT_x2)
		top = tf.maximum(pred_y1, GT_y1)
		bottom = tf.minimum(pred_y2, GT_y2)
		intersection = tf.maximum(tf.cast(0, tf.float32), right-left) * tf.maximum(tf.cast(0, tf.float32), bottom-top)
		union = area1 + area2 - intersection
		iou = intersection / union
		
		return iou
	
	def weights_init(self, shape, trainable, name):
		var = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(var, trainable=trainable, name=name)
	
	def biases_init(self, shape, trainable, name):
		var = tf.constant(0.1, shape=shape)
		return tf.Variable(var, trainable=trainable, name=name)
	
	def vgg16(self, input_tensor, weight_decay = 0.0005):
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
	
	def ROI_align(self, feature_map, boxes, box_num):
		rois = tf.image.crop_and_resize(feature_map, boxes = boxes, box_ind = [0 for _ in range(box_num)], crop_size = [14,14])
		return rois
	
	def FC_layer(self, net, out_num, weight_decay, stddev, name_post):
		regularizer = tf.contrib.layers.l2_regularizer(scale = weight_decay)
		weights = tf.get_variable(shape = [net.shape[-1], out_num], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=stddev), regularizer = regularizer, name = 'fc_weights_' + name_post)
		biases = tf.get_variable(shape = [out_num], dtype = tf.float32, initializer = tf.constant_initializer(0.0), name = 'fc_biases_' + name_post)
		output = tf.nn.bias_add(tf.matmul(net, weights), biases, name = 'fc_output_' + name_post)
		return output
	
	def faster_rcnn_head(self, feature_map, rois, image_h = 640, image_w = 640, RPN_k = 9, n_classes = 80, box_num = 24):
		'''
		PARAMS:
			anchor_x, anchor_y, anchor_w, anchor_h are input placeholders
			net_cls, net_reg are output of RPN network
			true_boxes_x, true_boxes_y, true_boxes_w, true_boxes_h, true_cls are ground truth bbox and class
		RETURN:
			net_cls_final: 		faster rcnn classification probs
			net_reg_final: 		faster rcnn bbox regression
			boxes_nms_indices: 		indices of region proposals for faster rcnn, after NMS
			boxes_y1_inside, boxes_x1_inside, boxes_y2_inside, boxes_x2_inside: 	(y1, x1, y2, x2) coords for all bboxes inside image range
		'''
		###########################################
		# generate 7x7 rois (inlcuding ROI align)
		rois = rois / image_h
		#return rois
		rois = self.ROI_align(feature_map, rois, box_num = box_num)
		#return rois
		rois = tf.layers.max_pooling2d(rois, pool_size = (2,2), strides = (2,2), padding = 'same', name = 'ROI_Align_pool')
		#return rois
		###########################################
		# Fast RCNN net
		with tf.variable_scope('faster_rcnn_head'):
			net = tf.contrib.layers.flatten(rois)
			#return net
			net = self.FC_layer(net, out_num = 4096, weight_decay = 0.0005, stddev = 0.01, name_post = '1')
			net = tf.nn.relu(net)
			#return net
			net = self.FC_layer(net, out_num = 4096, weight_decay = 0.0005, stddev = 0.01, name_post = '2')
			net = tf.nn.relu(net)
			#return net
			net_cls_final = self.FC_layer(net, out_num = n_classes+1, weight_decay = 0.0005, stddev = 0.01, name_post = 'cls_final')
			#net_cls_final = tf.nn.softmax(net_cls_final)
			net_reg_final = self.FC_layer(net, out_num = 4*n_classes, weight_decay = 0.0005, stddev = 0.001, name_post = 'reg_final')
		
		return net_cls_final, net_reg_final
