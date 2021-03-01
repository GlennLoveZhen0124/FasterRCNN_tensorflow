import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('./')
sys.path.append('../')
from models.backbones import resnet50
from models.necks import FPN


class RPN(resnet50.Backbone, FPN.FPN):
	def __init__(self, train_process_step):
		assert train_process_step in ['inference2', 'inference4']
		fpn_trainable_dict = {'P2': False, 'P3': False, 'P4': False, 'P5': False}
		resnet50.Backbone.__init__(self, train_process_step)
		FPN.FPN.__init__(self, fpn_trainable_dict)
	
	def RPN_head(self, x, reuse = False, RPN_k = 9):
		with tf.variable_scope('RPN_network', reuse = reuse):
			net = tf.layers.conv2d(x, filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu, trainable = False, name = 'RPN_conv3x3')
			net_cls = tf.layers.conv2d(net, filters = 2*RPN_k, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, trainable = False, name = 'RPN_cls_conv1x1')
			net_cls_shape = net_cls.get_shape().as_list()
			net_cls = tf.reshape(net_cls, (net_cls_shape[0],net_cls_shape[1],net_cls_shape[2]*RPN_k,net_cls_shape[3]//RPN_k))
			net_cls = tf.nn.softmax(net_cls)
			net_cls = tf.reshape(net_cls, net_cls_shape)
			net_reg = tf.layers.conv2d(net, filters = 4*RPN_k, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, trainable = False, name = 'RPN_reg_conv1x1')
		return net_cls, net_reg	# net_cls: (1, 37, 50, 24), net_reg: (1, 37, 50, 48)
	
	def decode_feature_map(self, net_cls, net_reg, RPN_k = 9):	# RPN_k = 12
		'''
		RPN_k = 3 if using FPN, 15 in total cause there are 5 levels
		'''
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
	
	def build_rpn(self, input_tensor):
		'''
		RPN_k = 3 is using FPN, 15 in total cause there are 5 levels
		'''
		C2, C3, C4, C5 = resnet50.Backbone.build_backbone(self, input_tensor)
		P2, P3, P4, P5, P6 = FPN.FPN.build_fpn(self, C2, C3, C4, C5)
		net_cls_level2, net_reg_level2 = self.RPN_head(P2, reuse = False, RPN_k = 3)
		net_cls_level3, net_reg_level3 = self.RPN_head(P3, reuse = True, RPN_k = 3)
		net_cls_level4, net_reg_level4 = self.RPN_head(P4, reuse = True, RPN_k = 3)
		net_cls_level5, net_reg_level5 = self.RPN_head(P5, reuse = True, RPN_k = 3)
		net_cls_level6, net_reg_level6 = self.RPN_head(P6, reuse = True, RPN_k = 3)
		net_reg_x_level2, net_reg_y_level2, net_reg_w_level2, net_reg_h_level2, net_cls_0_level2, net_cls_1_level2 = self.decode_feature_map(net_cls_level2, net_reg_level2, RPN_k = 3)
		net_reg_x_level3, net_reg_y_level3, net_reg_w_level3, net_reg_h_level3, net_cls_0_level3, net_cls_1_level3 = self.decode_feature_map(net_cls_level3, net_reg_level3, RPN_k = 3)
		net_reg_x_level4, net_reg_y_level4, net_reg_w_level4, net_reg_h_level4, net_cls_0_level4, net_cls_1_level4 = self.decode_feature_map(net_cls_level4, net_reg_level4, RPN_k = 3)
		net_reg_x_level5, net_reg_y_level5, net_reg_w_level5, net_reg_h_level5, net_cls_0_level5, net_cls_1_level5 = self.decode_feature_map(net_cls_level5, net_reg_level5, RPN_k = 3)
		net_reg_x_level6, net_reg_y_level6, net_reg_w_level6, net_reg_h_level6, net_cls_0_level6, net_cls_1_level6 = self.decode_feature_map(net_cls_level6, net_reg_level6, RPN_k = 3)
		ret = dict(level2 = (net_reg_x_level2, net_reg_y_level2, net_reg_w_level2, net_reg_h_level2, net_cls_0_level2, net_cls_1_level2), 
			level3 = (net_reg_x_level3, net_reg_y_level3, net_reg_w_level3, net_reg_h_level3, net_cls_0_level3, net_cls_1_level3),
			level4 = (net_reg_x_level4, net_reg_y_level4, net_reg_w_level4, net_reg_h_level4, net_cls_0_level4, net_cls_1_level4),
			level5 = (net_reg_x_level5, net_reg_y_level5, net_reg_w_level5, net_reg_h_level5, net_cls_0_level5, net_cls_1_level5), 
			level6 = (net_reg_x_level6, net_reg_y_level6, net_reg_w_level6, net_reg_h_level6, net_cls_0_level6, net_cls_1_level6)
			)
		return ret









