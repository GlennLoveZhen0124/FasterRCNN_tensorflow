from proposal_config import Config
from RPN_FPN_model import RPN
from RPN_FPN_data_multiprocess import Data
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


class Proposals(object):
	def __init__(self, args):
		self.train_process_step = args.train_process_step
		assert self.train_process_step in ['inference2', 'inference4']
		self.gpu_id = args.gpu_id
		os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		self.RPN_k = Config['RPN_k']
		self.n_classes = Config['n_classes']
		self.base_dir = Config['base_dir']
		self.train_data_folder = Config['train_data_folder']
		self.val_data_folder = Config['validate_data_folder']
		self.width, self.height = Config['width'], Config['height']
		self.strides = dict(level2 = 4, level3 = 8, level4 = 16, level5 = 32, level6 = 64)
		self.VM = RPN(self.train_process_step)
		self.data_train = Data(self.train_data_folder)
		self.data_val = Data(self.val_data_folder)
		self.x = tf.placeholder(tf.float32, [1, self.width * self.height * 3])
		self.input_tensor = tf.reshape(self.x, [1, self.width, self.height, 3])
		if self.train_process_step == 'inference2':
			self.train_proposal_save_folder = os.path.join(self.base_dir, 'dataset/train/proposals_step2/')
			self.val_proposal_save_folder = os.path.join(self.base_dir, 'dataset/val/proposals_step2/')
			self.check_and_create(self.train_proposal_save_folder)
			self.check_and_create(self.val_proposal_save_folder)
			self.backbone_model = os.path.join(self.base_dir, 'checkpoints/backbone_checkpoints/step1/backbone.ckpt-best')
			self.rpn_model = os.path.join(self.base_dir, 'checkpoints/RPN_checkpoints/step1/model_RPN.ckpt-best')
		elif self.train_process_step == 'inference4':
			self.train_proposal_save_folder = os.path.join(self.base_dir, 'dataset/train/proposals_step4/')
			self.val_proposal_save_folder = os.path.join(self.base_dir, 'dataset/val/proposals_step4/')
			self.check_and_create(self.train_proposal_save_folder)
			self.check_and_create(self.val_proposal_save_folder)
			self.backbone_model = os.path.join(self.base_dir, 'checkpoints/backbone_checkpoints/step3/backbone.ckpt-best')
			self.rpn_model = os.path.join(self.base_dir, 'checkpoints/RPN_checkpoints/step3/model_RPN.ckpt-best')
	
	def check_and_create(self, folder):
		if not os.path.exists(folder):
			os.makedirs(folder)
	
	def draw_rect(self, img, boxes):
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
	
	def foward_tensors(self):
		ret_dict = self.VM.build_rpn(self.input_tensor)	# all levels' RPN output (cls + reg), after decode
		return ret_dict
	
	def get_level_tensors(self, net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1, level):
		anchor_x_placeholder = tf.placeholder(tf.float32, [self.width//self.strides['level{}'.format(level)], self.height//self.strides['level{}'.format(level)], self.RPN_k])
		anchor_y_placeholder = tf.placeholder(tf.float32, [self.width//self.strides['level{}'.format(level)], self.height//self.strides['level{}'.format(level)], self.RPN_k])
		anchor_w_placeholder = tf.placeholder(tf.float32, [self.width//self.strides['level{}'.format(level)], self.height//self.strides['level{}'.format(level)], self.RPN_k])
		anchor_h_placeholder = tf.placeholder(tf.float32, [self.width//self.strides['level{}'.format(level)], self.height//self.strides['level{}'.format(level)], self.RPN_k])
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
		inside_y2_cond = tf.less(boxes_y2, tf.cast(self.height, tf.float32))
		inside_x2_cond = tf.less(boxes_x2, tf.cast(self.width, tf.float32))
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
		print('boxes: ', boxes)
		print('boxes_T: ', boxes_T)
		print('scores: ', scores)
		print('scores_inside: ', scores_inside)
		print('boxes_nms_indices: ', boxes_nms_indices)
		print('boxes_nms: ', boxes_nms)
		return boxes_T, scores_inside, anchor_x_placeholder, anchor_y_placeholder, anchor_w_placeholder, anchor_h_placeholder
	
	def generate_proposals(self, sess, tensors, data, data_folder, proposal_save_folder):
		boxes_T_level2, scores_inside_level2, anchor_x_placeholder_level2, anchor_y_placeholder_level2, anchor_w_placeholder_level2, anchor_h_placeholder_level2 = tensors['level2']
		boxes_T_level3, scores_inside_level3, anchor_x_placeholder_level3, anchor_y_placeholder_level3, anchor_w_placeholder_level3, anchor_h_placeholder_level3 = tensors['level3']
		boxes_T_level4, scores_inside_level4, anchor_x_placeholder_level4, anchor_y_placeholder_level4, anchor_w_placeholder_level4, anchor_h_placeholder_level4 = tensors['level4']
		boxes_T_level5, scores_inside_level5, anchor_x_placeholder_level5, anchor_y_placeholder_level5, anchor_w_placeholder_level5, anchor_h_placeholder_level5 = tensors['level5']
		boxes_T_level6, scores_inside_level6, anchor_x_placeholder_level6, anchor_y_placeholder_level6, anchor_w_placeholder_level6, anchor_h_placeholder_level6 = tensors['level6']
		boxes_T = tf.concat([boxes_T_level2, boxes_T_level3, boxes_T_level4, boxes_T_level5, boxes_T_level6], axis = 0)
		scores_inside = tf.concat([scores_inside_level2, scores_inside_level3, scores_inside_level4, scores_inside_level5, scores_inside_level6], axis = 0)
		boxes_nms_indices = tf.image.non_max_suppression(boxes = boxes_T, scores = scores_inside, max_output_size = 1000, iou_threshold = 0.7)	# here I change max_output_size from 300 to 1000
		boxes_nms = tf.gather(boxes_T, boxes_nms_indices)
		gen = data.get_batch(1)
		total = len(os.listdir(data_folder))//2
		for i in tqdm(range(total)):
			image_path, img, ret_level2, ret_level3, ret_level4, ret_level5, ret_level6 = gen.__next__()
			proposal_save_path = '.'.join(os.path.basename(image_path).split('.')[:-1]) + '.txt'
			proposal_save_path = os.path.join(proposal_save_folder, proposal_save_path)
			all_anchors_x_level2, all_anchors_y_level2, all_anchors_w_level2, all_anchors_h_level2 = ret_level2
			all_anchors_x_level3, all_anchors_y_level3, all_anchors_w_level3, all_anchors_h_level3 = ret_level3
			all_anchors_x_level4, all_anchors_y_level4, all_anchors_w_level4, all_anchors_h_level4 = ret_level4
			all_anchors_x_level5, all_anchors_y_level5, all_anchors_w_level5, all_anchors_h_level5 = ret_level5
			all_anchors_x_level6, all_anchors_y_level6, all_anchors_w_level6, all_anchors_h_level6 = ret_level6
			feed_dict = {
				self.x: img,
				anchor_x_placeholder_level2: all_anchors_x_level2,
				anchor_x_placeholder_level3: all_anchors_x_level3,
				anchor_x_placeholder_level4: all_anchors_x_level4,
				anchor_x_placeholder_level5: all_anchors_x_level5,
				anchor_x_placeholder_level6: all_anchors_x_level6,
				anchor_y_placeholder_level2: all_anchors_y_level2,
				anchor_y_placeholder_level3: all_anchors_y_level3,
				anchor_y_placeholder_level4: all_anchors_y_level4,
				anchor_y_placeholder_level5: all_anchors_y_level5,
				anchor_y_placeholder_level6: all_anchors_y_level6,
				anchor_w_placeholder_level2: all_anchors_w_level2,
				anchor_w_placeholder_level3: all_anchors_w_level3,
				anchor_w_placeholder_level4: all_anchors_w_level4,
				anchor_w_placeholder_level5: all_anchors_w_level5,
				anchor_w_placeholder_level6: all_anchors_w_level6,
				anchor_h_placeholder_level2: all_anchors_h_level2,
				anchor_h_placeholder_level3: all_anchors_h_level3,
				anchor_h_placeholder_level4: all_anchors_h_level4,
				anchor_h_placeholder_level5: all_anchors_h_level5,
				anchor_h_placeholder_level6: all_anchors_h_level6
			}
			boxes_nms_ret, scores_inside_ret = sess.run([boxes_nms, scores_inside], feed_dict = feed_dict)
			for bb in boxes_nms_ret:
				L = []
				for b in bb:
					L.append(str(b))
				s = ','.join(L)
				with open(proposal_save_path, 'a') as log:
					log.write(s)
					log.write('\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Generate proposals.')
	parser.add_argument('-step', '--train_process_step', type = str, choices = ['inference2','inference4'], help = 'process step number of training FasterRCNN.')
	parser.add_argument('-gpu', '--gpu_id', type = str, required = True, help = 'GPU id for use.')
	args = parser.parse_args()
	
	train_dataset_folder = Config['train_data_folder']
	val_dataset_folder = Config['validate_data_folder']
	PR = Proposals(args)
	ret_dict = PR.foward_tensors()
	net_reg_x_level2, net_reg_y_level2, net_reg_w_level2, net_reg_h_level2, net_cls_0_level2, net_cls_1_level2 = ret_dict['level2']
	net_reg_x_level3, net_reg_y_level3, net_reg_w_level3, net_reg_h_level3, net_cls_0_level3, net_cls_1_level3 = ret_dict['level3']
	net_reg_x_level4, net_reg_y_level4, net_reg_w_level4, net_reg_h_level4, net_cls_0_level4, net_cls_1_level4 = ret_dict['level4']
	net_reg_x_level5, net_reg_y_level5, net_reg_w_level5, net_reg_h_level5, net_cls_0_level5, net_cls_1_level5 = ret_dict['level5']
	net_reg_x_level6, net_reg_y_level6, net_reg_w_level6, net_reg_h_level6, net_cls_0_level6, net_cls_1_level6 = ret_dict['level6']
	
	boxes_T_level2, scores_inside_level2, anchor_x_placeholder_level2, anchor_y_placeholder_level2, anchor_w_placeholder_level2, anchor_h_placeholder_level2 = PR.get_level_tensors(net_reg_x_level2, net_reg_y_level2, net_reg_w_level2, net_reg_h_level2, net_cls_0_level2, net_cls_1_level2, level=2)
	boxes_T_level3, scores_inside_level3, anchor_x_placeholder_level3, anchor_y_placeholder_level3, anchor_w_placeholder_level3, anchor_h_placeholder_level3 = PR.get_level_tensors(net_reg_x_level3, net_reg_y_level3, net_reg_w_level3, net_reg_h_level3, net_cls_0_level3, net_cls_1_level3, level=3)
	boxes_T_level4, scores_inside_level4, anchor_x_placeholder_level4, anchor_y_placeholder_level4, anchor_w_placeholder_level4, anchor_h_placeholder_level4 = PR.get_level_tensors(net_reg_x_level4, net_reg_y_level4, net_reg_w_level4, net_reg_h_level4, net_cls_0_level4, net_cls_1_level4, level=4)
	boxes_T_level5, scores_inside_level5, anchor_x_placeholder_level5, anchor_y_placeholder_level5, anchor_w_placeholder_level5, anchor_h_placeholder_level5 = PR.get_level_tensors(net_reg_x_level5, net_reg_y_level5, net_reg_w_level5, net_reg_h_level5, net_cls_0_level5, net_cls_1_level5, level=5)
	boxes_T_level6, scores_inside_level6, anchor_x_placeholder_level6, anchor_y_placeholder_level6, anchor_w_placeholder_level6, anchor_h_placeholder_level6 = PR.get_level_tensors(net_reg_x_level6, net_reg_y_level6, net_reg_w_level6, net_reg_h_level6, net_cls_0_level6, net_cls_1_level6, level=6)
	
	tensors = dict(level2 = (boxes_T_level2, scores_inside_level2, anchor_x_placeholder_level2, anchor_y_placeholder_level2, anchor_w_placeholder_level2, anchor_h_placeholder_level2), level3 = (boxes_T_level3, scores_inside_level3, anchor_x_placeholder_level3, anchor_y_placeholder_level3, anchor_w_placeholder_level3, anchor_h_placeholder_level3), level4 = (boxes_T_level4, scores_inside_level4, anchor_x_placeholder_level4, anchor_y_placeholder_level4, anchor_w_placeholder_level4, anchor_h_placeholder_level4), level5 = (boxes_T_level5, scores_inside_level5, anchor_x_placeholder_level5, anchor_y_placeholder_level5, anchor_w_placeholder_level5, anchor_h_placeholder_level5), level6 = (boxes_T_level6, scores_inside_level6, anchor_x_placeholder_level6, anchor_y_placeholder_level6, anchor_w_placeholder_level6, anchor_h_placeholder_level6))
	boxes_T = tf.concat([boxes_T_level2, boxes_T_level3, boxes_T_level4, boxes_T_level5, boxes_T_level6], axis = 0)
	scores_inside = tf.concat([scores_inside_level2, scores_inside_level3, scores_inside_level4, scores_inside_level5, scores_inside_level6], axis = 0)
	
	global_vars = tf.global_variables()
	data_train = PR.data_train
	data_val = PR.data_val
	backbone_vars = [v for v in global_vars if 'resnet_v1_50' in v.name and 'global_step' not in v.name and 'mean_rgb' not in v.name and 'logits' not in v.name]
	rpn_fpn_vars = [v for v in global_vars if 'RPN_network' in v.name or 'FPN' in v.name]
	saver1 = tf.train.Saver(var_list = backbone_vars)
	saver2 = tf.train.Saver(var_list = rpn_fpn_vars)
	with tf.Session() as sess:
		saver1.restore(sess, PR.backbone_model)
		saver2.restore(sess, PR.rpn_model)
		PR.generate_proposals(sess, tensors, data_val, PR.val_data_folder, PR.val_proposal_save_folder)
		PR.generate_proposals(sess, tensors, data_train, PR.train_data_folder, PR.train_proposal_save_folder)


