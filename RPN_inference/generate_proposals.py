from proposal_config import Config
from RPN_model import VGGModel
from RPN_data import Data
from PIL import Image, ImageDraw
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import scipy.misc
import argparse
import random
import json
import cv2
import sys
import os


def draw_rect(img, boxes):
	pil_img = Image.fromarray(img)
	draw = ImageDraw.Draw(pil_img)
	for box in boxes:
		#box_x, box_y, box_w, box_h = box
		box_y1, box_x1, box_y2, box_x2 = box
		#left = int(box_x - box_w/2)
		left = int(box_x1)
		#right = int(box_x + box_w/2)
		right = int(box_x2)
		#top = int(box_y - box_h/2)
		top = int(box_y1)
		#bottom = int(box_y + box_h/2)
		bottom = int(box_y2)
		draw.rectangle(((left, top), (right, bottom)), outline=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
	del draw
	pil_img.save('xxx.jpg')

def check_and_create(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def generate_proposals(sess, tensors, data, data_folder, proposal_save_folder):
	net_cls, net_reg, net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1, boxes, boxes_nms, scores_inside = tensors
	gen = data.get_batch(1)
	total = len(os.listdir(data_folder))//2
	for i in tqdm(range(total)):
		image_path, img, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = gen.__next__()
		proposal_save_path = os.path.basename(image_path).split('.')[0] + '.txt'
		proposal_save_path = os.path.join(proposal_save_folder, proposal_save_path)
		cls, reg, reg_x, reg_y, reg_w, reg_h, cls0, cls1, boxes_ret, boxes_nms_ret, scores_inside_ret = sess.run([net_cls, net_reg, net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1, boxes, boxes_nms, scores_inside], feed_dict = {x: img, anchor_x_placeholder: all_anchors_x, anchor_y_placeholder: all_anchors_y, anchor_w_placeholder: all_anchors_w, anchor_h_placeholder: all_anchors_h})
		for bb in boxes_nms_ret:
			L = []
			for b in bb:
				L.append(str(b))
			s = ','.join(L)
			with open(proposal_save_path, 'a') as log:
				log.write(s)
				log.write('\n')


parser = argparse.ArgumentParser(description = 'Generate proposals.')
parser.add_argument('-step', '--train_process_step', type = int, choices = [2,4], help = 'process step number of training FasterRCNN.(4 in total)')
parser.add_argument('-gpu', '--gpu_id', type = str, required = True, help = 'GPU id for use.')
args = parser.parse_args()

train_process_step = args.train_process_step
assert train_process_step in [2,4]
gpu_id = args.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RPN_k = Config['RPN_k']
n_classes = Config['n_classes']
base_dir = Config['base_dir']
train_data_folder = Config['train_data_folder']
val_data_folder = Config['validate_data_folder']
width, height = Config['width'], Config['height']

VM = VGGModel()
data_train = Data(train_data_folder)
data_val = Data(val_data_folder)

x = tf.placeholder(tf.float32, [1, width * height * 3])
input_tensor = tf.reshape(x, [1, width, height, 3])

anchor_x_placeholder = tf.placeholder(tf.float32, [width//16, height//16, RPN_k])
anchor_y_placeholder = tf.placeholder(tf.float32, [width//16, height//16, RPN_k])
anchor_w_placeholder = tf.placeholder(tf.float32, [width//16, height//16, RPN_k])
anchor_h_placeholder = tf.placeholder(tf.float32, [width//16, height//16, RPN_k])

if train_process_step == 2:
	train_proposal_save_folder = os.path.join(base_dir, 'dataset/train/proposals_step2/')
	val_proposal_save_folder = os.path.join(base_dir, 'dataset/val/proposals_step2/')
	check_and_create(train_proposal_save_folder)
	check_and_create(val_proposal_save_folder)
	backbone_model = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step1/backbone.ckpt-best')
	rpn_model = os.path.join(base_dir, 'checkpoints/RPN_checkpoints/step1/model_RPN.ckpt-best')
elif train_process_step == 4:
	train_proposal_save_folder = os.path.join(base_dir, 'dataset/train/proposals_step4/')
	val_proposal_save_folder = os.path.join(base_dir, 'dataset/val/proposals_step4/')
	check_and_create(train_proposal_save_folder)
	check_and_create(val_proposal_save_folder)
	backbone_model = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step3/backbone.ckpt-best')
	rpn_model = os.path.join(base_dir, 'checkpoints/RPN_checkpoints/step3/model_RPN.ckpt-best')

feature_map = VM.vgg16(input_tensor)
net_cls, net_reg = VM.RPN_head(feature_map, n_classes, RPN_k = RPN_k)
net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1 = VM.decode_feature_map(net_cls, net_reg, RPN_k = RPN_k)
net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1 = tf.squeeze(net_reg_x), tf.squeeze(net_reg_y), tf.squeeze(net_reg_w), tf.squeeze(net_reg_h), tf.squeeze(net_cls_0), tf.squeeze(net_cls_1)

boxes_x = anchor_x_placeholder + net_reg_x
boxes_y = anchor_y_placeholder + net_reg_y
boxes_w = anchor_w_placeholder + net_reg_w
boxes_h = anchor_h_placeholder + net_reg_h

boxes_y1 = boxes_y - boxes_h/2
boxes_x1 = boxes_x - boxes_w/2
boxes_y2 = boxes_y + boxes_h/2
boxes_x2 = boxes_x + boxes_w/2
inside_y1_cond = tf.greater(boxes_y1, tf.cast(0, tf.float32))
inside_x1_cond = tf.greater(boxes_x1, tf.cast(0, tf.float32))
inside_y2_cond = tf.less(boxes_y2, tf.cast(height, tf.float32))
inside_x2_cond = tf.less(boxes_x2, tf.cast(width, tf.float32))
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

global_vars = tf.global_variables()
vgg_vars = [v for v in global_vars if 'vgg_16' in v.name]
rpn_vars = [v for v in global_vars if 'RPN_network' in v.name]
saver1 = tf.train.Saver(var_list = vgg_vars)
saver2 = tf.train.Saver(var_list = rpn_vars)

with tf.Session() as sess:
	saver1.restore(sess, backbone_model)
	saver2.restore(sess, rpn_model)
	tensors = [net_cls, net_reg, net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1, boxes, boxes_nms, scores_inside]
	generate_proposals(sess, tensors, data_train, train_data_folder, train_proposal_save_folder)
	generate_proposals(sess, tensors, data_val, val_data_folder, val_proposal_save_folder)
	


