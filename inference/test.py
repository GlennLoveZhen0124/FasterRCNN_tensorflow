from PIL import Image, ImageDraw
from FasterRCNN_model import FasterRCNNModel
from FasterRCNN_config import Config
from RPN_data import Data
import tensorflow as tf
import numpy as np
import scipy.misc
import argparse
import random
import json
import cv2
import sys
import os
import cpu_nms


def generate_proposals(net_cls_1, net_reg_x, net_reg_y, net_reg_w, net_reg_h, anchor_x, anchor_y, anchor_w, anchor_h):
	boxes_x = anchor_x + net_reg_x
	boxes_y = anchor_y + net_reg_y
	boxes_w = anchor_w + net_reg_w
	boxes_h = anchor_h + net_reg_h
	boxes_y1 = boxes_y - boxes_h/2
	boxes_x1 = boxes_x - boxes_w/2
	boxes_y2 = boxes_y + boxes_h/2
	boxes_x2 = boxes_x + boxes_w/2
	inside_y1_cond = tf.greater(boxes_y1, tf.cast(0, tf.float32))
	inside_x1_cond = tf.greater(boxes_x1, tf.cast(0, tf.float32))
	inside_y2_cond = tf.less(boxes_y2, tf.cast(640, tf.float32))
	inside_x2_cond = tf.less(boxes_x2, tf.cast(640, tf.float32))
	ind_inside = tf.where(tf.logical_and(tf.logical_and(tf.logical_and(inside_y1_cond, inside_x1_cond), inside_y2_cond), inside_x2_cond))
	boxes_y1_inside = tf.gather_nd(boxes_y1, ind_inside)
	boxes_x1_inside = tf.gather_nd(boxes_x1, ind_inside)
	boxes_y2_inside = tf.gather_nd(boxes_y2, ind_inside)
	boxes_x2_inside = tf.gather_nd(boxes_x2, ind_inside)
	
	boxes = tf.stack([tf.reshape(boxes_y1_inside,[-1]), tf.reshape(boxes_x1_inside, [-1]), tf.reshape(boxes_y2_inside, [-1]), tf.reshape(boxes_x2_inside, [-1])])
	boxes_T = tf.transpose(boxes)
	scores = tf.reshape(net_cls_1, [-1])
	scores_inside = tf.reshape(tf.gather_nd(net_cls_1, ind_inside), [-1])
	boxes_nms_indices = tf.image.non_max_suppression(boxes = boxes_T, scores = scores_inside, max_output_size = 300, iou_threshold = 0.7)
	boxes_nms = tf.gather(boxes_T, boxes_nms_indices)
	scores_nms = tf.gather(scores_inside, boxes_nms_indices)
	
	return boxes_nms_indices, boxes_nms, scores_nms

def invert_from_delta(pos_net_cls, pos_net_reg_y1, pos_net_reg_x1, pos_net_reg_y2, pos_net_reg_x2, proposals, image_h = 640, image_w = 640):
	pos_num = tf.shape(pos_net_cls)[0]
	ind_xs = tf.range(pos_num)
	max_pos_net_reg_index = tf.cast(tf.argmax(pos_net_cls[:,1:], axis = 1), tf.int32)
	max_inds = tf.stack([ind_xs, max_pos_net_reg_index])
	max_inds = tf.transpose(max_inds)
	pos_net_reg_y1_u = tf.gather_nd(pos_net_reg_y1, max_inds)
	pos_net_reg_x1_u = tf.gather_nd(pos_net_reg_x1, max_inds)
	pos_net_reg_y2_u = tf.gather_nd(pos_net_reg_y2, max_inds)
	pos_net_reg_x2_u = tf.gather_nd(pos_net_reg_x2, max_inds)
	
	pos_net_reg_x_u = (pos_net_reg_x1_u + pos_net_reg_x2_u) / 2
	pos_net_reg_y_u = (pos_net_reg_y1_u + pos_net_reg_y2_u) / 2
	pos_net_reg_w_u = pos_net_reg_x2_u - pos_net_reg_x1_u
	pos_net_reg_h_u = pos_net_reg_y2_u - pos_net_reg_y1_u
	
	proposal_x = (proposals[:,1] + proposals[:,3]) / 2
	proposal_y = (proposals[:,0] + proposals[:,2]) / 2
	proposal_w = proposals[:,3] - proposals[:,1]
	proposal_h = proposals[:,2] - proposals[:,0]
	
	GT_x_pred = image_w * pos_net_reg_x_u + proposal_x
	GT_y_pred = image_h * pos_net_reg_y_u + proposal_y
	GT_w_pred = tf.exp(pos_net_reg_w_u) * proposal_w
	GT_h_pred = tf.exp(pos_net_reg_h_u) * proposal_h
	
	return GT_x_pred, GT_y_pred, GT_w_pred, GT_h_pred

def get_class_names(class_name_file):
	with open(class_name_file) as f:
		data = f.readline().strip()
	jsondata = json.loads(data)
	return jsondata

def draw_func(img, cls_pred, reg_x, reg_y, reg_w, reg_h, class_index_to_name, save_path, image_h = 640, image_w = 640):
	pil_image = Image.fromarray(img)
	draw = ImageDraw.Draw(pil_image)
	for i in range(len(reg_x)):
		cls_ind = cls_pred[i]
		cls = class_index_to_name[cls_ind]
		x = reg_x[i]
		y = reg_y[i]
		w = reg_w[i]
		h = reg_h[i]
		left = int(x - w/2)
		if left < 0:
			left = 0
		right = int(x + w/2)
		if right > image_w:
			right = image_w
		top = int(y - h/2)
		if top < 0:
			top = 0
		bottom = int(y + h/2)
		if bottom > image_h:
			bottom = image_h
		draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
		draw.text((left + 6, bottom - 25), cls, fill=(255, 255, 255, 255))
	del draw
	pil_image.save(save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Inference code")
	parser.add_argument('-image', '--image_path_for_test', type = str, required = True, help = 'test image path')
	parser.add_argument('-save', '--save_path_for_draw_image', type = str, required = True, help = 'save path for draw rectangle image')
	parser.add_argument('-gpu', '--gpu_id', type = str, default = '0', help = 'GPU id for use.')
	args = parser.parse_args()
	
	image_path = args.image_path_for_test
	save_path = args.save_path_for_draw_image
	gpu_id = args.gpu_id
	
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	config = Config()
	class_name_file = config.class_name_file
	backbone_model_path = config.backbone_model_path
	RPN_model_path = config.RPN_model_path
	FastRCNN_model_path = config.FastRCNN_model_path
	image_h, image_w = config.image_h, config.image_w
	RPN_k = config.RPN_k
	n_classes = config.n_classes
	box_num = config.box_num
	class_name_to_index = get_class_names(class_name_file)
	class_index_to_name = {v:k for k,v in class_name_to_index.items()}
	RPN_data = Data()
	img, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = RPN_data.get_batch(image_path)
	img_for_draw = scipy.misc.imread(image_path)
	if img_for_draw.shape != (image_h, image_w,3):
		img_for_draw = scipy.misc.imresize(img_for_draw, (image_h, image_w))
	
	x = tf.placeholder(tf.float32, [1, image_h * image_w * 3])
	input_tensor = tf.reshape(x, [1, image_h, image_w, 3])
	anchor_x_placeholder = tf.placeholder(tf.float32, [image_w//16, image_h//16, RPN_k])
	anchor_y_placeholder = tf.placeholder(tf.float32, [image_w//16, image_h//16, RPN_k])
	anchor_w_placeholder = tf.placeholder(tf.float32, [image_w//16, image_h//16, RPN_k])
	anchor_h_placeholder = tf.placeholder(tf.float32, [image_w//16, image_h//16, RPN_k])
	VM = FasterRCNNModel()
	feature_map = VM.vgg16(input_tensor)
	net_cls, net_reg = VM.RPN_head(feature_map, n_classes, RPN_k=RPN_k)
	net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1 = VM.decode_feature_map(net_cls, net_reg, RPN_k = RPN_k)
	net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1 = tf.squeeze(net_reg_x), tf.squeeze(net_reg_y), tf.squeeze(net_reg_w), tf.squeeze(net_reg_h), tf.squeeze(net_cls_0), tf.squeeze(net_cls_1)
	boxes_nms_indices, boxes_nms, scores_nms = generate_proposals(net_cls_1, net_reg_x, net_reg_y, net_reg_w, net_reg_h, anchor_x_placeholder, anchor_y_placeholder, anchor_w_placeholder, anchor_h_placeholder)
	print('boxes_nms_indices: ', boxes_nms_indices)
	print('boxes_nms: ', boxes_nms)
	print('scores_nms: ', scores_nms)
	
	net_cls_final, net_reg_final = VM.faster_rcnn_head(feature_map, boxes_nms, image_h = image_h, image_w = image_w, RPN_k = RPN_k, n_classes = n_classes, box_num = box_num)
	max_net_cls_index = tf.argmax(net_cls_final, axis = 1)
	pos_net_cls_index = tf.where(tf.not_equal(max_net_cls_index, 0))
	pos_net_cls = tf.gather_nd(net_cls_final, pos_net_cls_index)		# (pos_num, 81)
	pos_net_reg = tf.gather_nd(net_reg_final, pos_net_cls_index)		# (pos_num, 320)
	pos_proposals = tf.gather_nd(boxes_nms, pos_net_cls_index)		# (pos_num, 4)
	
	pos_net_reg_shape = tf.shape(pos_net_reg)
	pos_net_reg_y1 = tf.strided_slice(pos_net_reg, [0,0], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	pos_net_reg_x1 = tf.strided_slice(pos_net_reg, [0,1], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	pos_net_reg_y2 = tf.strided_slice(pos_net_reg, [0,2], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	pos_net_reg_x2 = tf.strided_slice(pos_net_reg, [0,3], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	
	GT_x_pred, GT_y_pred, GT_w_pred, GT_h_pred = invert_from_delta(pos_net_cls, pos_net_reg_y1, pos_net_reg_x1, pos_net_reg_y2, pos_net_reg_x2, pos_proposals, image_h = image_h, image_w = image_w)
	cls_pred = tf.argmax(pos_net_cls, axis = 1)
	confidence = tf.nn.softmax(tf.reduce_max(pos_net_cls, axis = 1))
	
	######################################################################
	sess = tf.InteractiveSession()
	global_vars = tf.global_variables()
	vgg_vars = [v for v in global_vars if 'vgg_16' in v.name and 'Momentum' not in v.name]
	rpn_vars = [v for v in global_vars if 'RPN_network' in v.name and 'Momentum' not in v.name]
	fast_rcnn_vars = [v for v in global_vars if 'faster_rcnn_head' in v.name and 'Momentum' not in v.name]
	
	saver1 = tf.train.Saver(var_list = vgg_vars)
	saver2 = tf.train.Saver(var_list = rpn_vars)
	saver3 = tf.train.Saver(var_list = fast_rcnn_vars)
	
	saver1.restore(sess, backbone_model_path)
	saver2.restore(sess, RPN_model_path)
	saver3.restore(sess, FastRCNN_model_path)
	cls_pred_ret, GT_x_pred_ret, GT_y_pred_ret, GT_w_pred_ret, GT_h_pred_ret, confidence_ret = sess.run([cls_pred, GT_x_pred, GT_y_pred, GT_w_pred, GT_h_pred, confidence], feed_dict = {x: img, anchor_x_placeholder: all_anchors_x, anchor_y_placeholder: all_anchors_y, anchor_w_placeholder: all_anchors_w, anchor_h_placeholder: all_anchors_h})
	GT_y1 = GT_y_pred_ret - GT_h_pred_ret / 2
	GT_x1 = GT_x_pred_ret - GT_w_pred_ret / 2
	GT_y2 = GT_y_pred_ret + GT_h_pred_ret / 2
	GT_x2 = GT_x_pred_ret + GT_w_pred_ret / 2
	index_keep = cpu_nms.py_cpu_nms(GT_y1, GT_x1, GT_y2, GT_x2, confidence_ret, thresh = 0.5)
	GT_x_keep = GT_x_pred_ret[index_keep]
	GT_y_keep = GT_y_pred_ret[index_keep]
	GT_w_keep = GT_w_pred_ret[index_keep]
	GT_h_keep = GT_h_pred_ret[index_keep]
	cls_keep = cls_pred_ret[index_keep]
	draw_func(img_for_draw, cls_keep, GT_x_keep, GT_y_keep, GT_w_keep, GT_h_keep, class_index_to_name, save_path = save_path, image_h = image_h, image_w = image_w)
	print('confidence: ', confidence_ret, confidence_ret.shape)
	




