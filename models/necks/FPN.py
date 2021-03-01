import tensorflow as tf
import numpy as np
import os


class FPN(object):
	def __init__(self, trainable_dict_fpn):
		'''
		trainable_dict_fpn: {'P2': True, 'P3': True, 'P4': True, 'P5': True}
		'''
		self.trainable_dict_fpn = trainable_dict_fpn
	
	def weights_init_fpn(self, shape, trainable, name):
		var = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def biases_init_fpn(self, shape, trainable, name):
		var = tf.constant(0.1, shape=shape)
		return tf.Variable(var, trainable = trainable, name=name)
	
	def build_fpn(self,C2,C3,C4,C5):
		'''
		C2: stride = 4, C3: stride = 8, C4: stride = 16, C5: stride = 32
		'''
		with tf.variable_scope('FPN'):
			weights_p5 = self.weights_init_fpn(shape=[1, 1, 2048, 256], trainable=self.trainable_dict_fpn['P5'], name = 'weights_p5')
			P5 = tf.nn.conv2d(C5, weights_p5, strides = [1,1,1,1], padding = 'VALID', name = 'conv_p5')
			P5_shape = P5.get_shape()
			P6 = tf.image.resize(P5, size = (P5_shape[1]//2, P5_shape[2]//2), method = 'nearest')
			weights_p4 = self.weights_init_fpn(shape=[1, 1, 1024, 256], trainable=self.trainable_dict_fpn['P4'], name = 'weights_p4')
			P4_ = tf.nn.conv2d(C4, weights_p4, strides = [1,1,1,1], padding = 'VALID', name = 'conv_p4')
			weights_p3 = self.weights_init_fpn(shape=[1, 1, 512, 256], trainable=self.trainable_dict_fpn['P3'], name = 'weights_p3')
			P3_ = tf.nn.conv2d(C3, weights_p3, strides = [1,1,1,1], padding = 'VALID', name = 'conv_p3')
			weights_p2 = self.weights_init_fpn(shape=[1, 1, 256, 256], trainable=self.trainable_dict_fpn['P2'], name = 'weights_p2')
			P2_ = tf.nn.conv2d(C2, weights_p2, strides = [1,1,1,1], padding = 'VALID', name = 'conv_p2')
			size_P4 = tf.shape(P4_)[1:3]
			size_P3 = tf.shape(P3_)[1:3]
			size_P2 = tf.shape(P2_)[1:3]
			P4 = P4_ + tf.image.resize_images(P5, size_P4, method = 1)
			P3 = P3_ + tf.image.resize_images(P4, size_P3, method = 1)
			P2 = P2_ + tf.image.resize_images(P3, size_P2, method = 1)
			weights_p5_final = self.weights_init_fpn(shape=[3, 3, 256, 256], trainable=self.trainable_dict_fpn['P5'], name = 'weights_p5_final')
			P5 = tf.nn.conv2d(P5, weights_p5_final, strides = [1,1,1,1], padding = 'SAME', name = 'conv_p5_final')
			weights_p4_final = self.weights_init_fpn(shape=[3, 3, 256, 256], trainable=self.trainable_dict_fpn['P4'], name = 'weights_p4_final')
			P4 = tf.nn.conv2d(P4, weights_p4_final, strides = [1,1,1,1], padding = 'SAME', name = 'conv_p4_final')
			weights_p3_final = self.weights_init_fpn(shape=[3, 3, 256, 256], trainable=self.trainable_dict_fpn['P3'], name = 'weights_p3_final')
			P3 = tf.nn.conv2d(P3, weights_p3_final, strides = [1,1,1,1], padding = 'SAME', name = 'conv_p3_final')
			weights_p2_final = self.weights_init_fpn(shape=[3, 3, 256, 256], trainable=self.trainable_dict_fpn['P2'], name = 'weights_p2_final')
			P2 = tf.nn.conv2d(P2, weights_p2_final, strides = [1,1,1,1], padding = 'SAME', name = 'conv_p2_final')
		return P2,P3,P4,P5,P6










