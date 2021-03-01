import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('./')
sys.path.append('../')
from models.backbones import resnet50
from models.necks import FPN


class FasterRCNN(resnet50.Backbone, FPN.FPN):
	def __init__(self, train_process_step):
		assert train_process_step in [2,4]
		if train_process_step == 2:
			fpn_trainable_dict = {'P2': True, 'P3': True, 'P4': True, 'P5': True}
		elif train_process_step == 4:
			fpn_trainable_dict = {'P2': False, 'P3': False, 'P4': False, 'P5': False}
		resnet50.Backbone.__init__(self, train_process_step)
		FPN.FPN.__init__(self, fpn_trainable_dict)
	
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
	
	def ROI_align(self, feature_map, boxes, box_num):
		rois = tf.image.crop_and_resize(feature_map, boxes = boxes, box_ind = [0 for _ in range(box_num)], crop_size = [14,14])
		return rois
	
	def FC_layer(self, net, out_num, weight_decay, stddev, name_post):
		regularizer = tf.contrib.layers.l2_regularizer(scale = weight_decay)
		weights = tf.get_variable(shape = [net.shape[-1], out_num], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=stddev), regularizer = regularizer, name = 'fc_weights_' + name_post)
		biases = tf.get_variable(shape = [out_num], dtype = tf.float32, initializer = tf.constant_initializer(0.0), name = 'fc_biases_' + name_post)
		output = tf.nn.bias_add(tf.matmul(net, weights), biases, name = 'fc_output_' + name_post)
		return output
	
	def faster_rcnn_head(self, feature_map, rois, reuse = False, image_h = 640, image_w = 640, RPN_k = 9, n_classes = 80, box_num = 24):
		'''
		PARAMS:
			feature_map is the output of backbone
			rois are proposals placeholders
		RETURN:
			net_cls_final: 		faster rcnn classification probs
			net_reg_final: 		faster rcnn bbox regression
			boxes_nms_indices: 		indices of region proposals for faster rcnn, after NMS
			boxes_y1_inside, boxes_x1_inside, boxes_y2_inside, boxes_x2_inside: 	(y1, x1, y2, x2) coords for all bboxes inside image range
		'''
		###########################################
		# generate 7x7 rois (including ROI align)
		rois = rois / image_h
		#return rois
		rois = self.ROI_align(feature_map, rois, box_num = box_num)
		#return rois
		rois = tf.layers.max_pooling2d(rois, pool_size = (2,2), strides = (2,2), padding = 'same', name = 'ROI_Align_pool')
		#return rois
		###########################################
		# Fast RCNN net
		with tf.variable_scope('faster_rcnn_head', reuse = reuse):
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
	
	def build_fast_rcnn(self, input_tensor, rois, onehot, y1, x1, y2, x2, image_h = 640, image_w = 640, RPN_k = 3, n_classes = 80, box_num = 64):
		C2, C3, C4, C5 = resnet50.Backbone.build_backbone(self, input_tensor)
		P2, P3, P4, P5, P6 = FPN.FPN.build_fpn(self, C2, C3, C4, C5)
		net_cls_level2, net_reg_level2 = self.faster_rcnn_head(P2, rois, reuse = False, image_h = image_h, image_w = image_w, RPN_k = RPN_k, n_classes = n_classes, box_num = box_num)
		net_cls_level3, net_reg_level3 = self.faster_rcnn_head(P3, rois, reuse = True, image_h = image_h, image_w = image_w, RPN_k = RPN_k, n_classes = n_classes, box_num = box_num)
		net_cls_level4, net_reg_level4 = self.faster_rcnn_head(P4, rois, reuse = True, image_h = image_h, image_w = image_w, RPN_k = RPN_k, n_classes = n_classes, box_num = box_num)
		net_cls_level5, net_reg_level5 = self.faster_rcnn_head(P5, rois, reuse = True, image_h = image_h, image_w = image_w, RPN_k = RPN_k, n_classes = n_classes, box_num = box_num)
		ret = dict(level2 = (net_cls_level2, net_reg_level2), 
			level3 = (net_cls_level3, net_reg_level3),
			level4 = (net_cls_level4, net_reg_level4),
			level5 = (net_cls_level5, net_reg_level5)
		)
		
		ret_use = self.distribute_by_area(rois, ret, onehot, y1, x1, y2, x2)	# add this
		
		return ret_use		# return ret_use instead of ret
	
	def get_tensor_area(self, proposals):
		area = (proposals[..., 2] - proposals[..., 0]) * (proposals[..., 3] - proposals[..., 1])
		return area
	
	def distribute_by_area(self, rois, fast_rcnn_ret, onehot, y1, x1, y2, x2):
		'''
		PARAMS:
			rois are proposals placeholders
			fast_rcnn_ret are the returned tensor dict of build_fast_rcnn
		RETURN:
			selected tensors for each level
		
		According to the formulation:
			k = round(k0 + log(sqrt(w*h)/224 ,2))
		we have,
		level5: area >= 100353
		level4: 25087 < area < 100353
		level3: 6272< area <= 25087
		level2: area <= 6272
		'''
		
		areas = self.get_tensor_area(rois)
		area_index_level2 = tf.where(tf.less_equal(areas, 6272))
		area_index_level3 = tf.where(tf.logical_and(tf.greater(areas, 6272), tf.less_equal(areas, 25087)))
		area_index_level4 = tf.where(tf.logical_and(tf.greater(areas, 25087), tf.less(areas, 100353)))
		area_index_level5 = tf.where(tf.greater_equal(areas, 100353))
		
		net_cls_level2, net_reg_level2 = fast_rcnn_ret['level2']
		net_cls_level3, net_reg_level3 = fast_rcnn_ret['level3']
		net_cls_level4, net_reg_level4 = fast_rcnn_ret['level4']
		net_cls_level5, net_reg_level5 = fast_rcnn_ret['level5']
		
		rois_level2 = tf.gather_nd(rois, area_index_level2)
		rois_level3 = tf.gather_nd(rois, area_index_level3)
		rois_level4 = tf.gather_nd(rois, area_index_level4)
		rois_level5 = tf.gather_nd(rois, area_index_level5)
		
		net_cls_use_level2, net_reg_use_level2 = tf.gather_nd(net_cls_level2, area_index_level2), tf.gather_nd(net_reg_level2, area_index_level2)
		net_cls_use_level3, net_reg_use_level3 = tf.gather_nd(net_cls_level3, area_index_level3), tf.gather_nd(net_reg_level3, area_index_level3)
		net_cls_use_level4, net_reg_use_level4 = tf.gather_nd(net_cls_level4, area_index_level4), tf.gather_nd(net_reg_level4, area_index_level4)
		net_cls_use_level5, net_reg_use_level5 = tf.gather_nd(net_cls_level5, area_index_level5), tf.gather_nd(net_reg_level5, area_index_level5)
		
		onehot_level2 = tf.gather_nd(onehot, area_index_level2)
		onehot_level3 = tf.gather_nd(onehot, area_index_level3)
		onehot_level4 = tf.gather_nd(onehot, area_index_level4)
		onehot_level5 = tf.gather_nd(onehot, area_index_level5)
		
		y1_level2 = tf.gather_nd(y1, area_index_level2)
		y1_level3 = tf.gather_nd(y1, area_index_level3)
		y1_level4 = tf.gather_nd(y1, area_index_level4)
		y1_level5 = tf.gather_nd(y1, area_index_level5)
		
		x1_level2 = tf.gather_nd(x1, area_index_level2)
		x1_level3 = tf.gather_nd(x1, area_index_level3)
		x1_level4 = tf.gather_nd(x1, area_index_level4)
		x1_level5 = tf.gather_nd(x1, area_index_level5)
		
		y2_level2 = tf.gather_nd(y2, area_index_level2)
		y2_level3 = tf.gather_nd(y2, area_index_level3)
		y2_level4 = tf.gather_nd(y2, area_index_level4)
		y2_level5 = tf.gather_nd(y2, area_index_level5)
		
		x2_level2 = tf.gather_nd(x2, area_index_level2)
		x2_level3 = tf.gather_nd(x2, area_index_level3)
		x2_level4 = tf.gather_nd(x2, area_index_level4)
		x2_level5 = tf.gather_nd(x2, area_index_level5)
		
		ret = dict(level2 = (net_cls_use_level2, net_reg_use_level2, rois_level2, onehot_level2, y1_level2, x1_level2, y2_level2, x2_level2),
			level3 = (net_cls_use_level3, net_reg_use_level3, rois_level3, onehot_level3, y1_level3, x1_level3, y2_level3, x2_level3),
			level4 = (net_cls_use_level4, net_reg_use_level4, rois_level4, onehot_level4, y1_level4, x1_level4, y2_level4, x2_level4),
			level5 = (net_cls_use_level5, net_reg_use_level5, rois_level5, onehot_level5, y1_level5, x1_level5, y2_level5, x2_level5)
		)
		
		return ret













