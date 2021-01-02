import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from Faster_RCNN_data import Data
from Faster_RCNN_utils import Utils
from Faster_RCNN_losses import Losses
from Faster_RCNN_config import Config
from Faster_RCNN_model import FasterRCNNModel


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Train Fast RCNN code, including step2 and step4.")
	parser.add_argument('-step', '--train_process_step', type = int, choices = [2,4], help = 'process step number of training FasterRCNN.(4 in total)')
	parser.add_argument('-pretrain', '--pretrain_model', type = str, default = '', help = 'pretrain model basename, exp, vgg_16.ckpt')
	parser.add_argument('-backbone', '--backbone_checkpoint', type = str, default = 'backbone.ckpt-best', help = 'backbone best checkpoint basename of step3.')
	parser.add_argument('-fastrcnn', '--fastrcnn_checkpoint', type = str, default = 'fast_rcnn.ckpt-best', help = 'fast rcnn bestonly checkpoint basename of step2.')
	parser.add_argument('-gpu', '--gpu_id', type = str, default = '0', help = 'GPU id for training.')
	args = parser.parse_args()
	
	base_dir = Config['base_dir']
	train_steps = Config['train_steps']
	init_lr = Config['learning_rate_init']
	decay_steps = Config['decay_steps']
	decay_rate = Config['decay_rate']
	stair_case = Config['stair_case']
	train_folder = Config['train_data_folder']
	val_folder = Config['validate_data_folder']
	class_name_file = Config['class_name_file']
	n_classes = Config['n_classes']
	
	train_process_step = args.train_process_step
	pretrain_model = args.pretrain_model
	backbone_checkpoint = args.backbone_checkpoint
	fast_rcnn_checkpoint = args.fastrcnn_checkpoint
	gpu_id = args.gpu_id
	
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	log_folder = os.path.join(base_dir, 'logs')
	
	if train_process_step == 2:
		backbone_pretrain_model_path = os.path.join(base_dir, 'checkpoints/backbone_pretrain_checkpoints/{}'.format(pretrain_model))
		step2_backbone_save_folder = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step2/')
		step2_fast_rcnn_save_folder = os.path.join(base_dir, 'checkpoints/Fast_RCNN_checkpoints/step2/')
		step2_backbone_save_path = os.path.join(step2_backbone_save_folder, 'backbone.ckpt')
		step2_fast_rcnn_save_path = os.path.join(step2_fast_rcnn_save_folder, 'fast_rcnn.ckpt')
		train_proposals_folder = os.path.join(base_dir, 'dataset/train/proposals_step2/')
		val_proposals_folder = os.path.join(base_dir, 'dataset/val/proposals_step2/')
		train_log = os.path.join(log_folder, 'train_FastRCNN_step2.log')
		best_loss_log = os.path.join(log_folder, 'best_loss_FastRCNN_step2.log')
	
	elif train_process_step == 4:
		backbone_pretrain_model_path = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step3/' + backbone_checkpoint)
		fast_rcnn_pretrain_model_path = os.path.join(base_dir, 'checkpoints/Fast_RCNN_checkpoints/step2/' + fast_rcnn_checkpoint)
		step4_backbone_save_folder =os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step4/')
		step4_fast_rcnn_save_folder = os.path.join(base_dir, 'checkpoints/Fast_RCNN_checkpoints/step4/')
		step4_backbone_save_path = os.path.join(step4_backbone_save_folder, 'backbone.ckpt')
		step4_fast_rcnn_save_path = os.path.join(step4_fast_rcnn_save_folder, 'fast_rcnn.ckpt')
		train_proposals_folder = os.path.join(base_dir, 'dataset/train/proposals_step4/')
		val_proposals_folder = os.path.join(base_dir, 'dataset/val/proposals_step4/')
		train_log = os.path.join(log_folder, 'train_FastRCNN_step4.log')
		best_loss_log = os.path.join(log_folder, 'best_loss_FastRCNN_step4.log')
		
	
	width, height = 640, 640
	RPN_k = 12
	nms_num = 300
	global_step = tf.Variable(0, trainable = False)
	lr = tf.train.exponential_decay(learning_rate = init_lr, global_step = global_step, decay_steps = decay_steps, decay_rate = decay_rate, staircase = stair_case)
	mu = 0.9
	batch_size = 64
	utils = Utils()
	data = Data(train_folder, train_proposals_folder, class_name_file)
	data_val = Data(val_folder, val_proposals_folder, class_name_file)
	gen = data.get_batch(batch_size)
	gen_val = data_val.get_batch(batch_size)
	
	x = tf.placeholder(tf.float32, [1, width * height * 3])
	input_tensor = tf.reshape(x, [-1, height, width, 3])
	proposals_placeholder = tf.placeholder(tf.float32, [batch_size, 4])		# 4 represents for [y1, x1, y2, x2]
	onehot_placeholder = tf.placeholder(tf.float32, [batch_size, n_classes+1])
	box_max_y1_placeholder = tf.placeholder(tf.float32, [batch_size])
	box_max_x1_placeholder = tf.placeholder(tf.float32, [batch_size])
	box_max_y2_placeholder = tf.placeholder(tf.float32, [batch_size])
	box_max_x2_placeholder = tf.placeholder(tf.float32, [batch_size])
	
	VM = FasterRCNNModel(train_process_step)
	LS = Losses()
	feature_map = VM.vgg16(input_tensor)
	net_cls_final, net_reg_final = VM.faster_rcnn_head(feature_map, proposals_placeholder, image_h = height, image_w = width, RPN_k = RPN_k, n_classes = n_classes, box_num = batch_size)
	classification_loss, regression_loss = LS.loss_layer(net_cls_final, net_reg_final, onehot_placeholder, box_max_y1_placeholder, box_max_x1_placeholder, box_max_y2_placeholder, box_max_x2_placeholder, proposals_placeholder, image_w = 640, image_h = 640)
	total_loss = classification_loss + regression_loss
	train_op = tf.train.MomentumOptimizer(learning_rate = lr, momentum = mu).minimize(total_loss, global_step=global_step)
	
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	global_vars = tf.global_variables()
	vgg_vars = [v for v in global_vars if 'vgg_16' in v.name and 'Momentum' not in v.name]
	fast_rcnn_vars = [v for v in global_vars if 'faster_rcnn_head' in v.name and 'Momentum' not in v.name]
	
	saver1 = tf.train.Saver(var_list = vgg_vars)
	saver2 = tf.train.Saver(var_list = fast_rcnn_vars)
	saver1.restore(sess, backbone_pretrain_model_path)
	if train_process_step == 4:
		saver2.restore(sess, fast_rcnn_pretrain_model_path)
	best_loss = float('inf')
	for step in range(train_steps+1):
		batch_xml, img, proposals_in_batch, classes_in_batch_onehot, box_max_y1_in_batch, box_max_x1_in_batch, box_max_y2_in_batch, box_max_x2_in_batch, pos_num = gen.__next__()
		batch_xml_val, img_val, proposals_in_batch_val, classes_in_batch_onehot_val, box_max_y1_in_batch_val, box_max_x1_in_batch_val, box_max_y2_in_batch_val, box_max_x2_in_batch_val, pos_num_val = gen_val.__next__()
		cls_loss_ret, reg_loss_ret, total_loss_ret, lr_ret, _ = sess.run([classification_loss, regression_loss, total_loss, lr, train_op], feed_dict = {x: img, onehot_placeholder: classes_in_batch_onehot, box_max_y1_placeholder: box_max_y1_in_batch, box_max_x1_placeholder: box_max_x1_in_batch, box_max_y2_placeholder: box_max_y2_in_batch, box_max_x2_placeholder: box_max_x2_in_batch, proposals_placeholder: proposals_in_batch})
		cls_loss_ret_val, reg_loss_ret_val, total_loss_ret_val = sess.run([classification_loss, regression_loss, total_loss], feed_dict = {x: img_val, onehot_placeholder: classes_in_batch_onehot_val, box_max_y1_placeholder: box_max_y1_in_batch_val, box_max_x1_placeholder: box_max_x1_in_batch_val, box_max_y2_placeholder: box_max_y2_in_batch_val, box_max_x2_placeholder: box_max_x2_in_batch_val, proposals_placeholder: proposals_in_batch_val})
		with open(train_log, 'a') as log:
			log.write('Batch {}: LearningRate: {}, Total_train_loss = {}, cls_train_loss = {}, reg_train_loss = {}, Total_val_loss = {}, cls_val_loss = {}, reg_val_loss = {}'.format(step, lr_ret, total_loss_ret, cls_loss_ret, reg_loss_ret, total_loss_ret_val, cls_loss_ret_val, reg_loss_ret_val))
			log.write('\n')
		if step != 0 and step % 100 == 0:
			if train_process_step == 2:
				saver1.save(sess, step2_backbone_save_path, global_step = step)
				saver2.save(sess, step2_fast_rcnn_save_path, global_step = step)
			elif train_process_step == 4:
				saver1.save(sess, step4_backbone_save_path, global_step = step)
				saver2.save(sess, step4_fast_rcnn_save_path, global_step = step)
			if total_loss_ret_val < best_loss:
				best_loss = total_loss_ret_val
				if train_process_step == 2:
					utils.copy_files(str(step), 'best', step2_backbone_save_folder)
					utils.copy_files(str(step), 'best', step2_fast_rcnn_save_folder)
				elif train_process_step == 4:
					utils.copy_files(str(step), 'best', step4_backbone_save_folder)
					utils.copy_files(str(step), 'best', step4_fast_rcnn_save_folder)
				with open(best_loss_log, 'a') as log:
					log.write('model: {}, loss: {}'.format(str(step), str(total_loss_ret_val)))
					log.write('\n')
					









