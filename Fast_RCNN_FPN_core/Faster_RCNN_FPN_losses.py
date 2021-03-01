import tensorflow as tf
import numpy as np


class Losses(object):
	def __init__(self):
		pass
	
	def smoothL1(self, d, net_reg):
		diff = tf.cast(tf.abs(d - net_reg), tf.float32)
		#diff = tf.cast(tf.abs(d), tf.float32)
		less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
		loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
		return tf.reduce_mean(loss)
	
	def convert_to_delta(self, GT_y1, GT_x1, GT_y2, GT_x2, proposals, image_w = 640, image_h = 640):
		GT_x = (GT_x1 + GT_x2) / 2
		GT_y = (GT_y1 + GT_y2) / 2
		GT_w = GT_x2 - GT_x1
		GT_h = GT_y2 - GT_y1
		proposal_x = (proposals[:,1] + proposals[:,3]) / 2
		proposal_y = (proposals[:,0] + proposals[:,2]) / 2
		proposal_w = proposals[:,3] - proposals[:,1]
		proposal_h = proposals[:,2] - proposals[:,0]
		dx = (GT_x - proposal_x) / image_w
		dy = (GT_y - proposal_y) / image_h
		dw = tf.log(GT_w/proposal_w)
		dh = tf.log(GT_h/proposal_h)
		return dx, dy, dw, dh
	
	def cls_loss(self, net_cls_final, onehot_label):
		cond = tf.greater(tf.shape(net_cls_final)[0], 0)
		#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = onehot_label, logits = net_cls_final))
		loss = tf.cond(cond, lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = onehot_label, logits = net_cls_final)), lambda: 0.0)
		return loss
	
	def reg_loss(self, net_reg_final, pos_onehot_label, box_max_y1_in_batch, box_max_x1_in_batch, box_max_y2_in_batch, box_max_x2_in_batch, proposals, image_w = 640, image_h = 640):
		'''
		when there are some positive proposals in batch, reg loss will go in here to be computed, otherwise, reg loss wont go here
		net_reg_final: [pos_num, 320]
		pos_onehot_label: GT onehot label for ROI, [num_pos, 81]
		'''
		
		#net_reg_shape = net_reg_final.get_shape().as_list()
		net_reg_shape = tf.shape(net_reg_final)
		
		net_reg_y1 = tf.strided_slice(net_reg_final, [0,0], [net_reg_shape[0], net_reg_shape[1]], [1,4])
		net_reg_x1 = tf.strided_slice(net_reg_final, [0,1], [net_reg_shape[0], net_reg_shape[1]], [1,4])
		net_reg_y2 = tf.strided_slice(net_reg_final, [0,2], [net_reg_shape[0], net_reg_shape[1]], [1,4])
		net_reg_x2 = tf.strided_slice(net_reg_final, [0,3], [net_reg_shape[0], net_reg_shape[1]], [1,4])
		net_reg_x = (net_reg_x1 + net_reg_x2) / 2
		net_reg_y = (net_reg_y1 + net_reg_y2) / 2
		net_reg_w = net_reg_x2 - net_reg_x1
		net_reg_h = net_reg_y2 - net_reg_y1
		
		dx, dy, dw, dh = self.convert_to_delta(box_max_y1_in_batch, box_max_x1_in_batch, box_max_y2_in_batch, box_max_x2_in_batch, proposals)
		
		pos_onehot_label = pos_onehot_label[:, 1:]		# delete background , [pos_num, 81] => [pos_num, 80]
		pos_num = tf.shape(pos_onehot_label)[0]
		ind_xs = tf.range(pos_num)
		max_ind = tf.cast(tf.argmax(pos_onehot_label, axis = 1), tf.int32)
		max_inds = tf.stack([ind_xs, max_ind])
		max_inds = tf.transpose(max_inds)
		net_reg_x_u = tf.gather_nd(net_reg_x, max_inds)		# net_reg_x for true class u
		net_reg_y_u = tf.gather_nd(net_reg_y, max_inds)		# net_reg_y for true class u
		net_reg_w_u = tf.gather_nd(net_reg_w, max_inds)		# net_reg_w for true class u
		net_reg_h_u = tf.gather_nd(net_reg_h, max_inds)		# net_reg_h for true class u
		
		smoothL1_x = self.smoothL1(dx, net_reg_x_u)
		smoothL1_y = self.smoothL1(dy, net_reg_y_u)
		smoothL1_w = self.smoothL1(dw, net_reg_w_u)
		smoothL1_h = self.smoothL1(dh, net_reg_h_u)
		
		return smoothL1_x + smoothL1_y + smoothL1_w + smoothL1_h
	
	def loss_layer(self, net_cls_final, net_reg_final, onehot_label, box_max_y1_in_batch, box_max_x1_in_batch, box_max_y2_in_batch, box_max_x2_in_batch, proposals, image_w = 640, image_h = 640):
		'''
		net_cls_final: output of network for classification	(batch_size, 81)
		onehot_label: GT onehot label for classification	(batch_size, 81)
		net_reg_final: output of network for regression	(batch_size, 320)
		box_max_y1_in_batch: GT y1 coords for all examples	(batch_size, )
		box_max_x1_in_batch: GT x1 coords for all examples	(batch_size, )
		box_max_y2_in_batch: GT y2 coords for all examples	(batch_size, )
		box_max_x2_in_batch: GT x2 coords for all examples	(batch_size, )
		proposal_x: proposal x coords for all examples	(batch_size, )
		proposal_y: proposal y coords for all examples	(batch_size, )
		proposal_w: proposal w coords for all examples	(batch_size, )
		proposal_h: proposal h coords for all examples	(batch_size, )
		'''
		# need to change this method into multi-levels, each level get one out, do this method level times
		# use as: 
		#	classification_loss_level2, regression_loss_level2 = loss_layer(net_cls_level2, net_reg_level2, onehot_label_level2, box_max_y1_level2, box_max_x1_level2, box_max_y2_level2, box_max_x2_level2, proposals_level2, image_w=640, image_h = 640)
		#	classification_loss_level3, regression_loss_level3 = ... ...
		#	... ...
		
		
		max_inds = tf.argmax(onehot_label, axis = 1)
		pos_inds = tf.where(tf.not_equal(max_inds, 0))
		positive_onehot_label = tf.gather_nd(onehot_label, pos_inds)
		positive_shape = tf.shape(positive_onehot_label)[0]
		positive_reg = tf.gather_nd(net_reg_final, pos_inds)
		positive_proposals = tf.gather_nd(proposals, pos_inds)
		positive_box_max_y1_in_batch = tf.gather_nd(box_max_y1_in_batch, pos_inds)
		positive_box_max_x1_in_batch = tf.gather_nd(box_max_x1_in_batch, pos_inds)
		positive_box_max_y2_in_batch = tf.gather_nd(box_max_y2_in_batch, pos_inds)
		positive_box_max_x2_in_batch = tf.gather_nd(box_max_x2_in_batch, pos_inds)
		classification_loss = self.cls_loss(net_cls_final, onehot_label)
		regression_loss = tf.cond(tf.greater(positive_shape, 0), lambda : self.reg_loss(positive_reg, positive_onehot_label, positive_box_max_y1_in_batch, positive_box_max_x1_in_batch, positive_box_max_y2_in_batch, positive_box_max_x2_in_batch, positive_proposals), lambda : 0.0)
		
		return classification_loss, regression_loss, net_cls_final, onehot_label

