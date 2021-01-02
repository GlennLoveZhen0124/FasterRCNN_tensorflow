import os
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf
from RPN_config import Config
from RPN_data import Data
from RPN_model import VGGModel
from RPN_utils import Utils
from RPN_losses import Losses


if __name__ == '__main__':
	'''
	240k iterations for learning rate 0.001, 80k iterations for learning rate 0.0001
	saver1 restore vgg pretrain model
	saver1 save vgg fine-tuned variables
	saver2 save rpn variables
	'''
	
	parser = argparse.ArgumentParser(description = "Train RPN code, including step1 and step3.")
	parser.add_argument('-step', '--train_process_step', type = int, choices = [1,3], help = 'process step number of training FasterRCNN.(4 in total)')
	parser.add_argument('-pretrain', '--pretrain_model', type = str, default = '', help = 'pretrain model basename, exp, vgg_16.ckpt')
	parser.add_argument('-backbone', '--backbone_checkpoint', type = str, default = 'backbone.ckpt-best', help = 'backbone bestonly checkpoint basename of step2')
	parser.add_argument('-rpn', '--rpn_checkpoint', type = str, default = 'model_RPN.ckpt-best', help = 'rpn bestonly checkpoint basename of step1')
	parser.add_argument('-gpu', '--gpu_id', type = str, default = '0', help = 'GPU id for training.')
	args = parser.parse_args()
	
	base_dir = Config['base_dir']
	train_data_folder = Config['train_data_folder']
	validate_data_folder = Config['validate_data_folder']
	train_steps = Config['train_steps']
	init_lr = Config['learning_rate_init']
	decay_steps = Config['decay_steps']
	decay_rate = Config['decay_rate']
	stair_case = Config['stair_case']
	
	train_process_step = args.train_process_step
	pretrain_model = args.pretrain_model
	backbone_checkpoint = args.backbone_checkpoint
	rpn_checkpoint = args.rpn_checkpoint
	gpu_id = args.gpu_id
	assert train_process_step in [1,3]
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	log_folder = os.path.join(base_dir, 'logs')
	
	if train_process_step == 1:
		backbone_pretrain_model_path = os.path.join(base_dir, 'checkpoints/backbone_pretrain_checkpoints/{}'.format(pretrain_model))
		step1_backbone_save_folder = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step1/')
		step1_rpn_save_folder = os.path.join(base_dir, 'checkpoints/RPN_checkpoints/step1/')
		step1_backbone_save_path = os.path.join(step1_backbone_save_folder, 'backbone.ckpt')
		step1_rpn_save_path = os.path.join(step1_rpn_save_folder, 'model_RPN.ckpt')
		train_log = os.path.join(log_folder, 'train_RPN_step1.log')
		best_loss_log = os.path.join(log_folder, 'best_loss_RPN_step1.log')
	
	elif train_process_step == 3:
		backbone_pretrain_model_path = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step2/' + backbone_checkpoint)
		rpn_pretrain_model_path = os.path.join(base_dir, 'checkpoints/RPN_checkpoints/step1/' + rpn_checkpoint)
		step3_backbone_save_folder = os.path.join(base_dir, 'checkpoints/backbone_checkpoints/step3/')
		step3_rpn_save_folder = os.path.join(base_dir, 'checkpoints/RPN_checkpoints/step3/')
		step3_backbone_save_path = os.path.join(step3_backbone_save_folder, 'backbone.ckpt')
		step3_rpn_save_path = os.path.join(step3_rpn_save_folder, 'model_RPN.ckpt')
		train_log = os.path.join(log_folder, 'train_RPN_step3.log')
		best_loss_log = os.path.join(log_folder, 'best_loss_RPN_step3.log')
	
	
	data = Data(train_data_folder)
	data_val = Data(validate_data_folder)
	VM = VGGModel(train_process_step)
	Loss = Losses()
	utils = Utils()
	lmbd = 10
	#init_lr = 0.001
	global_step = tf.Variable(0, trainable=False, name = 'global_step')
	lr = tf.train.exponential_decay(learning_rate = init_lr, global_step = global_step, decay_steps = decay_steps, decay_rate = decay_rate, staircase = stair_case)
	
	x = tf.placeholder(tf.float32, [1, 640 * 640 * 3])
	input_tensor = tf.reshape(x, [1, 640, 640, 3])
	cls_labels = tf.placeholder(tf.int32, [40, 40, 12])
	all_anchors_x_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_anchors_y_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_anchors_w_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_anchors_h_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_gt_related_x_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_gt_related_y_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_gt_related_w_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	all_gt_related_h_placeholder = tf.placeholder(tf.float32, [40, 40, 12])
	
	gen = data.get_batch_anchors(256)
	gen_val = data_val.get_batch_anchors(256)
	
	vgg16_logit = VM.vgg16(input_tensor)
	net_cls, net_reg = VM.RPN_head(vgg16_logit, RPN_k = 12)
	net_reg_x, net_reg_y, net_reg_w, net_reg_h, net_cls_0, net_cls_1 = VM.decode_feature_map(net_cls, net_reg, RPN_k = 12)
	classification_loss = Loss.cls_loss(cls_labels, net_cls_0, net_cls_1)
	#batch_tx, batch_ty, batch_tw, batch_th, batch_tx_star, batch_ty_star, batch_tw_star, batch_th_star = utils.coords_convertion(all_labels, net_reg_x, net_reg_y, net_reg_w, net_reg_h, all_gt_related_x, all_gt_related_y, all_gt_related_w, all_gt_related_h, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h)
	batch_tx, batch_ty, batch_tw, batch_th, batch_tx_star, batch_ty_star, batch_tw_star, batch_th_star = utils.coords_convertion(cls_labels, net_reg_x, net_reg_y, net_reg_w, net_reg_h, all_gt_related_x_placeholder, all_gt_related_y_placeholder, all_gt_related_w_placeholder, all_gt_related_h_placeholder, all_anchors_x_placeholder, all_anchors_y_placeholder, all_anchors_w_placeholder, all_anchors_h_placeholder)
	regression_loss = Loss.reg_loss(batch_tx, batch_ty, batch_tw, batch_th, batch_tx_star, batch_ty_star, batch_tw_star, batch_th_star)
	#total_loss = lmbd * (1/(all_labels.shape[0]*all_labels.shape[1])) * tf.cast(regression_loss, tf.float32) + classification_loss
	total_loss = lmbd * (1/float(40*40)) * tf.cast(regression_loss, tf.float32) + classification_loss
	
	train_op = tf.train.GradientDescentOptimizer(learning_rate = lr, name = 'optimizer').minimize(total_loss, global_step = global_step)
	global_vars = tf.global_variables()
	vgg_var_list = [v for v in global_vars if 'vgg_16' in v.name and 'mean_rgb' not in v.name]
	rpn_var_list = [v for v in global_vars if 'RPN_network' in v.name]
	saver1 = tf.train.Saver(var_list = vgg_var_list)
	saver2 = tf.train.Saver(var_list = rpn_var_list)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver1.restore(sess, backbone_pretrain_model_path)
		if train_process_step == 3:
			saver2.restore(sess, rpn_pretrain_model_path)
		best_loss = float('inf')
		for step in range(train_steps+1):
			#print('===============================')
			image_path, img, all_labels, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h, all_gt_related_x, all_gt_related_y, all_gt_related_w, all_gt_related_h = gen.__next__()
			image_path_val, img_val, all_labels_val, all_anchors_x_val, all_anchors_y_val, all_anchors_w_val, all_anchors_h_val, all_gt_related_x_val, all_gt_related_y_val, all_gt_related_w_val, all_gt_related_h_val = gen_val.__next__()
			LOSS,cls_LOSS, reg_LOSS, _ = sess.run([total_loss, classification_loss, regression_loss, train_op], feed_dict = {x: img, cls_labels: all_labels, all_anchors_x_placeholder: all_anchors_x, all_anchors_y_placeholder: all_anchors_y, all_anchors_w_placeholder: all_anchors_w, all_anchors_h_placeholder: all_anchors_h, all_gt_related_x_placeholder: all_gt_related_x, all_gt_related_y_placeholder: all_gt_related_y, all_gt_related_w_placeholder: all_gt_related_w, all_gt_related_h_placeholder: all_gt_related_h})
			LOSS_val,cls_LOSS_val, reg_LOSS_val = sess.run([total_loss, classification_loss, regression_loss], feed_dict = {x: img_val, cls_labels: all_labels_val, all_anchors_x_placeholder: all_anchors_x_val, all_anchors_y_placeholder: all_anchors_y_val, all_anchors_w_placeholder: all_anchors_w_val, all_anchors_h_placeholder: all_anchors_h_val, all_gt_related_x_placeholder: all_gt_related_x_val, all_gt_related_y_placeholder: all_gt_related_y_val, all_gt_related_w_placeholder: all_gt_related_w_val, all_gt_related_h_placeholder: all_gt_related_h_val})
			#print('LOSS: ', LOSS)
			#print('LOSS_val: ', LOSS_val)
			with open(train_log, 'a') as log:
				log.write('Batch {}: Total_train_loss = {}, cls_train_loss = {}, reg_train_loss = {}, Total_val_loss = {}, cls_val_loss = {}, reg_val_loss = {}'.format(step, LOSS, cls_LOSS, reg_LOSS, LOSS_val, cls_LOSS_val, reg_LOSS_val))
				log.write('\n')
			if step % 100 == 0 and step != 0:
				if train_process_step == 1:
					saver1.save(sess, step1_backbone_save_path, global_step = step)
					saver2.save(sess, step1_rpn_save_path, global_step = step)
				elif train_process_step == 3:
					saver1.save(sess, step3_backbone_save_path, global_step = step)
					saver2.save(sess, step3_rpn_save_path, global_step = step)
				if LOSS_val < best_loss:
					best_loss = LOSS_val
					if train_process_step == 1:
						utils.copy_files(str(step), 'best', step1_backbone_save_folder)
						utils.copy_files(str(step), 'best', step1_rpn_save_folder)
					elif train_process_step == 3:
						utils.copy_files(str(step), 'best', step3_backbone_save_folder)
						utils.copy_files(str(step), 'best', step3_rpn_save_folder)
					with open(best_loss_log, 'a') as log:
						log.write('model: {}, loss: {}'.format(str(step), str(LOSS_val)))
						log.write('\n')








