import tensorflow as tf
import numpy as np
import scipy.misc
import argparse
import json
import sys
import cv2
import os
from my_Faster_RCNN_data import Data
from my_Faster_RCNN_model import FasterRCNNModel
from PIL import Image, ImageDraw


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

def get_proposals(proposal_path):
	proposals = []
	with open(proposal_path) as f:
		line = f.readline().strip()
		while line:
			y1, x1, y2, x2 = line.split(',')
			y1, x1, y2, x2 = float(y1), float(x1), float(y2), float(x2)
			proposals.append([y1,x1,y2,x2])
			line = f.readline().strip()
	proposals = np.array(proposals)
	return proposals

def get_COCO_names(COCO_name_file):
	with open(COCO_name_file) as f:
		data = f.readline().strip()
	jsondata = json.loads(data)
	return jsondata

def draw_func(img, cls_pred, reg_x, reg_y, reg_w, reg_h, COCO_index_to_name, save_path, image_h = 640, image_w = 640):
	pil_image = Image.fromarray(img)
	draw = ImageDraw.Draw(pil_image)
	for i in range(len(reg_x)):
		cls_ind = cls_pred[i]
		cls = COCO_index_to_name[cls_ind]
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
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'
	
	image_path = sys.argv[1]
	save_path = sys.argv[2]
	proposal_path = sys.argv[3]
	img = cv2.imread(image_path)
	if img.shape != (640,640,3):
		img = cv2.resize(img, (640,640))
	img = img / 255.
	img = img.flatten()
	img = img[np.newaxis, :]
	img_for_draw = scipy.misc.imread(image_path)
	proposals = get_proposals(proposal_path)
	COCO_name_file = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup/dataset/COCO_names.names'
	vgg_model_path = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup/checkpoints/VGG_checkpoints/checkpoints_bestonly/step4/vgg_16.ckpt-293200'
	fast_rcnn_model_path = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup/checkpoints/Fast_RCNN_checkpoints/checkpoints_bestonly/step4/fast_rcnn.ckpt-293200'
	COCO_name_to_index = get_COCO_names(COCO_name_file)
	COCO_index_to_name = {v:k for k,v in COCO_name_to_index.items()}
	image_h, image_w = 640, 640
	RPN_k = 12
	n_classes = 80
	box_num = 300
	x = tf.placeholder(tf.float32, [1, image_h*image_w*3])
	proposals_placeholder = tf.placeholder(tf.float32, [box_num, 4])	# (300, 4)
	input_tensor = tf.reshape(x, [1, image_h, image_w, 3])
	VM = FasterRCNNModel(4)
	feature_map = VM.vgg16(input_tensor)
	# net_cls_final : (300, 81),  net_reg_final : (300, 320)
	net_cls_final, net_reg_final = VM.faster_rcnn_head(feature_map, proposals_placeholder, image_h = image_h, image_w = image_w, RPN_k = RPN_k, n_classes = n_classes, box_num = box_num)
	max_net_cls_index = tf.argmax(net_cls_final, axis = 1)
	pos_net_cls_index = tf.where(tf.not_equal(max_net_cls_index, 0))
	pos_net_cls = tf.gather_nd(net_cls_final, pos_net_cls_index)		# (pos_num, 81)
	pos_net_reg = tf.gather_nd(net_reg_final, pos_net_cls_index)		# (pos_num, 320)
	pos_proposals = tf.gather_nd(proposals_placeholder, pos_net_cls_index)		# (pos_num, 4)
	
	pos_net_reg_shape = tf.shape(pos_net_reg)
	pos_net_reg_y1 = tf.strided_slice(pos_net_reg, [0,0], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	pos_net_reg_x1 = tf.strided_slice(pos_net_reg, [0,1], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	pos_net_reg_y2 = tf.strided_slice(pos_net_reg, [0,2], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	pos_net_reg_x2 = tf.strided_slice(pos_net_reg, [0,3], [pos_net_reg_shape[0], pos_net_reg_shape[1]], [1,4])	# (pos_num, 80)
	
	GT_x_pred, GT_y_pred, GT_w_pred, GT_h_pred = invert_from_delta(pos_net_cls, pos_net_reg_y1, pos_net_reg_x1, pos_net_reg_y2, pos_net_reg_x2, pos_proposals, image_h = image_h, image_w = image_w)
	cls_pred = tf.argmax(pos_net_cls, axis = 1)
	
	sess = tf.InteractiveSession()
	global_vars = tf.global_variables()
	vgg_vars = [v for v in global_vars if 'vgg_16' in v.name and 'Momentum' not in v.name]
	fast_rcnn_vars = [v for v in global_vars if 'faster_rcnn_head' in v.name and 'Momentum' not in v.name]
	saver1 = tf.train.Saver(var_list = vgg_vars)
	saver2 = tf.train.Saver(var_list = fast_rcnn_vars)
	saver1.restore(sess, vgg_model_path)
	saver2.restore(sess, fast_rcnn_model_path)
	
	cls_pred_ret, GT_x_pred_ret, GT_y_pred_ret, GT_w_pred_ret, GT_h_pred_ret = sess.run([cls_pred, GT_x_pred, GT_y_pred, GT_w_pred, GT_h_pred], feed_dict = {x: img, proposals_placeholder: proposals})
	draw_func(img_for_draw, cls_pred_ret, GT_x_pred_ret, GT_y_pred_ret, GT_w_pred_ret, GT_h_pred_ret, COCO_index_to_name, save_path = save_path, image_h = image_h, image_w = image_w)





