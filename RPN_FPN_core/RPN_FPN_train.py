import tensorflow as tf
import numpy as np
from RPN_FPN_data import Data
from RPN_FPN_model import RPN
from RPN_FPN_config import Config
from RPN_FPN_utils import Utils
from RPN_FPN_losses import Losses
import argparse
import os

'''
arr = np.ones((32,640,640,3))
arr = arr.astype(np.float32)
train_process_step = 1
rpn = RPN(train_process_step)
rpn_dict = rpn.build(arr)
print(rpn_dict)
'''

if __name__ == '__main__':
	'''
	240k iterations for learning rate 0.001, 80k iterations for learning rate 0.0001
	saver1 restore backbone pretrain model
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
	image_H = Config['image_H']
	image_W = Config['image_W']
	strides = Config['strides']
	
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
	#VM = VGGModel(train_process_step)
	VM = RPN(train_process_step)
	
	Loss = Losses()
	utils = Utils()
	lmbd = 10
	#init_lr = 0.001
	global_step = tf.Variable(0, trainable=False, name = 'global_step')
	lr = tf.train.exponential_decay(learning_rate = init_lr, global_step = global_step, decay_steps = decay_steps, decay_rate = decay_rate, staircase = stair_case)
	
	x = tf.placeholder(tf.float32, [1, image_W * image_H * 3])
	input_tensor = tf.reshape(x, [1, image_W, image_H, 3])
	
	cls_labels_level2 = tf.placeholder(tf.int32, [image_W//strides[0], image_H//strides[0], 3])
	all_anchors_x_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_anchors_y_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_anchors_w_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_anchors_h_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_gt_related_x_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_gt_related_y_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_gt_related_w_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	all_gt_related_h_placeholder_level2 = tf.placeholder(tf.float32, [image_W//strides[0], image_H//strides[0], 3])
	
	cls_labels_level3 = tf.placeholder(tf.int32, [image_W//strides[1], image_H//strides[1], 3])
	all_anchors_x_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_anchors_y_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_anchors_w_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_anchors_h_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_gt_related_x_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_gt_related_y_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_gt_related_w_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	all_gt_related_h_placeholder_level3 = tf.placeholder(tf.float32, [image_W//strides[1], image_H//strides[1], 3])
	
	cls_labels_level4 = tf.placeholder(tf.int32, [image_W//strides[2], image_H//strides[2], 3])
	all_anchors_x_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_anchors_y_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_anchors_w_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_anchors_h_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_gt_related_x_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_gt_related_y_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_gt_related_w_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	all_gt_related_h_placeholder_level4 = tf.placeholder(tf.float32, [image_W//strides[2], image_H//strides[2], 3])
	
	cls_labels_level5 = tf.placeholder(tf.int32, [image_W//strides[3], image_H//strides[3], 3])
	all_anchors_x_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_anchors_y_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_anchors_w_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_anchors_h_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_gt_related_x_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_gt_related_y_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_gt_related_w_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	all_gt_related_h_placeholder_level5 = tf.placeholder(tf.float32, [image_W//strides[3], image_H//strides[3], 3])
	
	cls_labels_level6 = tf.placeholder(tf.int32, [image_W//strides[4], image_H//strides[4], 3])
	all_anchors_x_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_anchors_y_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_anchors_w_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_anchors_h_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_gt_related_x_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_gt_related_y_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_gt_related_w_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	all_gt_related_h_placeholder_level6 = tf.placeholder(tf.float32, [image_W//strides[4], image_H//strides[4], 3])
	
	gen = data.get_batch_anchors(64)
	gen_val = data_val.get_batch_anchors(64)
	
	ret_dict = VM.build_rpn(input_tensor)	# all levels' RPN output (cls + reg), after decode
	net_reg_x_level2, net_reg_y_level2, net_reg_w_level2, net_reg_h_level2, net_cls_0_level2, net_cls_1_level2 = ret_dict['level2']
	net_reg_x_level3, net_reg_y_level3, net_reg_w_level3, net_reg_h_level3, net_cls_0_level3, net_cls_1_level3 = ret_dict['level3']
	net_reg_x_level4, net_reg_y_level4, net_reg_w_level4, net_reg_h_level4, net_cls_0_level4, net_cls_1_level4 = ret_dict['level4']
	net_reg_x_level5, net_reg_y_level5, net_reg_w_level5, net_reg_h_level5, net_cls_0_level5, net_cls_1_level5 = ret_dict['level5']
	net_reg_x_level6, net_reg_y_level6, net_reg_w_level6, net_reg_h_level6, net_cls_0_level6, net_cls_1_level6 = ret_dict['level6']
	
	classification_loss_level2 = Loss.cls_loss(cls_labels_level2, net_cls_0_level2, net_cls_1_level2)
	classification_loss_level3 = Loss.cls_loss(cls_labels_level3, net_cls_0_level3, net_cls_1_level3)
	classification_loss_level4 = Loss.cls_loss(cls_labels_level4, net_cls_0_level4, net_cls_1_level4)
	classification_loss_level5 = Loss.cls_loss(cls_labels_level5, net_cls_0_level5, net_cls_1_level5)
	classification_loss_level6 = Loss.cls_loss(cls_labels_level6, net_cls_0_level6, net_cls_1_level6)
	
	batch_tx_level2, batch_ty_level2, batch_tw_level2, batch_th_level2, batch_tx_star_level2, batch_ty_star_level2, batch_tw_star_level2, batch_th_star_level2 = utils.coords_convertion(cls_labels_level2, net_reg_x_level2, net_reg_y_level2, net_reg_w_level2, net_reg_h_level2, all_gt_related_x_placeholder_level2, all_gt_related_y_placeholder_level2, all_gt_related_w_placeholder_level2, all_gt_related_h_placeholder_level2, all_anchors_x_placeholder_level2, all_anchors_y_placeholder_level2, all_anchors_w_placeholder_level2, all_anchors_h_placeholder_level2)
	batch_tx_level3, batch_ty_level3, batch_tw_level3, batch_th_level3, batch_tx_star_level3, batch_ty_star_level3, batch_tw_star_level3, batch_th_star_level3 = utils.coords_convertion(cls_labels_level3, net_reg_x_level3, net_reg_y_level3, net_reg_w_level3, net_reg_h_level3, all_gt_related_x_placeholder_level3, all_gt_related_y_placeholder_level3, all_gt_related_w_placeholder_level3, all_gt_related_h_placeholder_level3, all_anchors_x_placeholder_level3, all_anchors_y_placeholder_level3, all_anchors_w_placeholder_level3, all_anchors_h_placeholder_level3)
	batch_tx_level4, batch_ty_level4, batch_tw_level4, batch_th_level4, batch_tx_star_level4, batch_ty_star_level4, batch_tw_star_level4, batch_th_star_level4 = utils.coords_convertion(cls_labels_level4, net_reg_x_level4, net_reg_y_level4, net_reg_w_level4, net_reg_h_level4, all_gt_related_x_placeholder_level4, all_gt_related_y_placeholder_level4, all_gt_related_w_placeholder_level4, all_gt_related_h_placeholder_level4, all_anchors_x_placeholder_level4, all_anchors_y_placeholder_level4, all_anchors_w_placeholder_level4, all_anchors_h_placeholder_level4)
	batch_tx_level5, batch_ty_level5, batch_tw_level5, batch_th_level5, batch_tx_star_level5, batch_ty_star_level5, batch_tw_star_level5, batch_th_star_level5 = utils.coords_convertion(cls_labels_level5, net_reg_x_level5, net_reg_y_level5, net_reg_w_level5, net_reg_h_level5, all_gt_related_x_placeholder_level5, all_gt_related_y_placeholder_level5, all_gt_related_w_placeholder_level5, all_gt_related_h_placeholder_level5, all_anchors_x_placeholder_level5, all_anchors_y_placeholder_level5, all_anchors_w_placeholder_level5, all_anchors_h_placeholder_level5)
	batch_tx_level6, batch_ty_level6, batch_tw_level6, batch_th_level6, batch_tx_star_level6, batch_ty_star_level6, batch_tw_star_level6, batch_th_star_level6 = utils.coords_convertion(cls_labels_level6, net_reg_x_level6, net_reg_y_level6, net_reg_w_level6, net_reg_h_level6, all_gt_related_x_placeholder_level6, all_gt_related_y_placeholder_level6, all_gt_related_w_placeholder_level6, all_gt_related_h_placeholder_level6, all_anchors_x_placeholder_level6, all_anchors_y_placeholder_level6, all_anchors_w_placeholder_level6, all_anchors_h_placeholder_level6)
	
	regression_loss_level2 = Loss.reg_loss(batch_tx_level2, batch_ty_level2, batch_tw_level2, batch_th_level2, batch_tx_star_level2, batch_ty_star_level2, batch_tw_star_level2, batch_th_star_level2)
	regression_loss_level3 = Loss.reg_loss(batch_tx_level3, batch_ty_level3, batch_tw_level3, batch_th_level3, batch_tx_star_level3, batch_ty_star_level3, batch_tw_star_level3, batch_th_star_level3)
	regression_loss_level4 = Loss.reg_loss(batch_tx_level4, batch_ty_level4, batch_tw_level4, batch_th_level4, batch_tx_star_level4, batch_ty_star_level4, batch_tw_star_level4, batch_th_star_level4)
	regression_loss_level5 = Loss.reg_loss(batch_tx_level5, batch_ty_level5, batch_tw_level5, batch_th_level5, batch_tx_star_level5, batch_ty_star_level5, batch_tw_star_level5, batch_th_star_level5)
	regression_loss_level6 = Loss.reg_loss(batch_tx_level6, batch_ty_level6, batch_tw_level6, batch_th_level6, batch_tx_star_level6, batch_ty_star_level6, batch_tw_star_level6, batch_th_star_level6)
	
	total_loss_level2 = lmbd * (1/float((image_W//strides[0])*(image_H//strides[0]))) * tf.cast(regression_loss_level2, tf.float32) + classification_loss_level2
	total_loss_level3 = lmbd * (1/float((image_W//strides[1])*(image_H//strides[1]))) * tf.cast(regression_loss_level3, tf.float32) + classification_loss_level3
	total_loss_level4 = lmbd * (1/float((image_W//strides[2])*(image_H//strides[2]))) * tf.cast(regression_loss_level4, tf.float32) + classification_loss_level4
	total_loss_level5 = lmbd * (1/float((image_W//strides[3])*(image_H//strides[3]))) * tf.cast(regression_loss_level5, tf.float32) + classification_loss_level5
	total_loss_level6 = lmbd * (1/float((image_W//strides[4])*(image_H//strides[4]))) * tf.cast(regression_loss_level6, tf.float32) + classification_loss_level6
	
	total_loss = total_loss_level2 + total_loss_level3 + total_loss_level4 + total_loss_level5 + total_loss_level6
	
	train_op = tf.train.GradientDescentOptimizer(learning_rate = lr, name = 'optimizer').minimize(total_loss, global_step = global_step)
	
	global_vars = tf.global_variables()
	backbone_var_list = [v for v in global_vars if 'resnet_v1_50' in v.name and 'global_step' not in v.name and 'mean_rgb' not in v.name and 'logits' not in v.name]
	rpn_fpn_var_list = [v for v in global_vars if 'RPN_network' in v.name or 'FPN' in v.name]
	saver1 = tf.train.Saver(var_list = backbone_var_list)
	saver2 = tf.train.Saver(var_list = rpn_fpn_var_list)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver1.restore(sess, backbone_pretrain_model_path)
		if train_process_step == 3:
			saver2.restore(sess, rpn_pretrain_model_path)
		best_loss = float('inf')
		for step in range(train_steps+1):
			image_path, img, ret_level2, ret_level3, ret_level4, ret_level5, ret_level6 = gen.__next__()
			image_path_val, img_val, ret_level2_val, ret_level3_val, ret_level4_val, ret_level5_val, ret_level6_val = gen_val.__next__()
			
			all_labels_level2, all_anchors_x_level2, all_anchors_y_level2, all_anchors_w_level2, all_anchors_h_level2, all_gt_related_x_level2, all_gt_related_y_level2, all_gt_related_w_level2, all_gt_related_h_level2 = ret_level2
			all_labels_level3, all_anchors_x_level3, all_anchors_y_level3, all_anchors_w_level3, all_anchors_h_level3, all_gt_related_x_level3, all_gt_related_y_level3, all_gt_related_w_level3, all_gt_related_h_level3 = ret_level3
			all_labels_level4, all_anchors_x_level4, all_anchors_y_level4, all_anchors_w_level4, all_anchors_h_level4, all_gt_related_x_level4, all_gt_related_y_level4, all_gt_related_w_level4, all_gt_related_h_level4 = ret_level4
			all_labels_level5, all_anchors_x_level5, all_anchors_y_level5, all_anchors_w_level5, all_anchors_h_level5, all_gt_related_x_level5, all_gt_related_y_level5, all_gt_related_w_level5, all_gt_related_h_level5 = ret_level5
			all_labels_level6, all_anchors_x_level6, all_anchors_y_level6, all_anchors_w_level6, all_anchors_h_level6, all_gt_related_x_level6, all_gt_related_y_level6, all_gt_related_w_level6, all_gt_related_h_level6 = ret_level6
			
			all_labels_level2_val, all_anchors_x_level2_val, all_anchors_y_level2_val, all_anchors_w_level2_val, all_anchors_h_level2_val, all_gt_related_x_level2_val, all_gt_related_y_level2_val, all_gt_related_w_level2_val, all_gt_related_h_level2_val = ret_level2_val
			all_labels_level3_val, all_anchors_x_level3_val, all_anchors_y_level3_val, all_anchors_w_level3_val, all_anchors_h_level3_val, all_gt_related_x_level3_val, all_gt_related_y_level3_val, all_gt_related_w_level3_val, all_gt_related_h_level3_val = ret_level3_val
			all_labels_level4_val, all_anchors_x_level4_val, all_anchors_y_level4_val, all_anchors_w_level4_val, all_anchors_h_level4_val, all_gt_related_x_level4_val, all_gt_related_y_level4_val, all_gt_related_w_level4_val, all_gt_related_h_level4_val = ret_level4_val
			all_labels_level5_val, all_anchors_x_level5_val, all_anchors_y_level5_val, all_anchors_w_level5_val, all_anchors_h_level5_val, all_gt_related_x_level5_val, all_gt_related_y_level5_val, all_gt_related_w_level5_val, all_gt_related_h_level5_val = ret_level5_val
			all_labels_level6_val, all_anchors_x_level6_val, all_anchors_y_level6_val, all_anchors_w_level6_val, all_anchors_h_level6_val, all_gt_related_x_level6_val, all_gt_related_y_level6_val, all_gt_related_w_level6_val, all_gt_related_h_level6_val = ret_level6_val
			feed_dict = {x:img, 
				cls_labels_level2: all_labels_level2,
				cls_labels_level3: all_labels_level3,
				cls_labels_level4: all_labels_level4,
				cls_labels_level5: all_labels_level5,
				cls_labels_level6: all_labels_level6,
				all_anchors_x_placeholder_level2: all_anchors_x_level2,
				all_anchors_x_placeholder_level3: all_anchors_x_level3,
				all_anchors_x_placeholder_level4: all_anchors_x_level4,
				all_anchors_x_placeholder_level5: all_anchors_x_level5,
				all_anchors_x_placeholder_level6: all_anchors_x_level6,
				all_anchors_y_placeholder_level2: all_anchors_y_level2,
				all_anchors_y_placeholder_level3: all_anchors_y_level3,
				all_anchors_y_placeholder_level4: all_anchors_y_level4,
				all_anchors_y_placeholder_level5: all_anchors_y_level5,
				all_anchors_y_placeholder_level6: all_anchors_y_level6,
				all_anchors_w_placeholder_level2: all_anchors_w_level2,
				all_anchors_w_placeholder_level3: all_anchors_w_level3,
				all_anchors_w_placeholder_level4: all_anchors_w_level4,
				all_anchors_w_placeholder_level5: all_anchors_w_level5,
				all_anchors_w_placeholder_level6: all_anchors_w_level6,
				all_anchors_h_placeholder_level2: all_anchors_h_level2,
				all_anchors_h_placeholder_level3: all_anchors_h_level3,
				all_anchors_h_placeholder_level4: all_anchors_h_level4,
				all_anchors_h_placeholder_level5: all_anchors_h_level5,
				all_anchors_h_placeholder_level6: all_anchors_h_level6,
				all_gt_related_x_placeholder_level2: all_gt_related_x_level2,
				all_gt_related_x_placeholder_level3: all_gt_related_x_level3,
				all_gt_related_x_placeholder_level4: all_gt_related_x_level4,
				all_gt_related_x_placeholder_level5: all_gt_related_x_level5,
				all_gt_related_x_placeholder_level6: all_gt_related_x_level6,
				all_gt_related_y_placeholder_level2: all_gt_related_y_level2,
				all_gt_related_y_placeholder_level3: all_gt_related_y_level3,
				all_gt_related_y_placeholder_level4: all_gt_related_y_level4,
				all_gt_related_y_placeholder_level5: all_gt_related_y_level5,
				all_gt_related_y_placeholder_level6: all_gt_related_y_level6,
				all_gt_related_w_placeholder_level2: all_gt_related_w_level2,
				all_gt_related_w_placeholder_level3: all_gt_related_w_level3,
				all_gt_related_w_placeholder_level4: all_gt_related_w_level4,
				all_gt_related_w_placeholder_level5: all_gt_related_w_level5,
				all_gt_related_w_placeholder_level6: all_gt_related_w_level6,
				all_gt_related_h_placeholder_level2: all_gt_related_h_level2,
				all_gt_related_h_placeholder_level3: all_gt_related_h_level3,
				all_gt_related_h_placeholder_level4: all_gt_related_h_level4,
				all_gt_related_h_placeholder_level5: all_gt_related_h_level5,
				all_gt_related_h_placeholder_level6: all_gt_related_h_level6
				}
			feed_dict_val = {x:img_val, 
				cls_labels_level2: all_labels_level2_val,
				cls_labels_level3: all_labels_level3_val,
				cls_labels_level4: all_labels_level4_val,
				cls_labels_level5: all_labels_level5_val,
				cls_labels_level6: all_labels_level6_val,
				all_anchors_x_placeholder_level2: all_anchors_x_level2_val,
				all_anchors_x_placeholder_level3: all_anchors_x_level3_val,
				all_anchors_x_placeholder_level4: all_anchors_x_level4_val,
				all_anchors_x_placeholder_level5: all_anchors_x_level5_val,
				all_anchors_x_placeholder_level6: all_anchors_x_level6_val,
				all_anchors_y_placeholder_level2: all_anchors_y_level2_val,
				all_anchors_y_placeholder_level3: all_anchors_y_level3_val,
				all_anchors_y_placeholder_level4: all_anchors_y_level4_val,
				all_anchors_y_placeholder_level5: all_anchors_y_level5_val,
				all_anchors_y_placeholder_level6: all_anchors_y_level6_val,
				all_anchors_w_placeholder_level2: all_anchors_w_level2_val,
				all_anchors_w_placeholder_level3: all_anchors_w_level3_val,
				all_anchors_w_placeholder_level4: all_anchors_w_level4_val,
				all_anchors_w_placeholder_level5: all_anchors_w_level5_val,
				all_anchors_w_placeholder_level6: all_anchors_w_level6_val,
				all_anchors_h_placeholder_level2: all_anchors_h_level2_val,
				all_anchors_h_placeholder_level3: all_anchors_h_level3_val,
				all_anchors_h_placeholder_level4: all_anchors_h_level4_val,
				all_anchors_h_placeholder_level5: all_anchors_h_level5_val,
				all_anchors_h_placeholder_level6: all_anchors_h_level6_val,
				all_gt_related_x_placeholder_level2: all_gt_related_x_level2_val,
				all_gt_related_x_placeholder_level3: all_gt_related_x_level3_val,
				all_gt_related_x_placeholder_level4: all_gt_related_x_level4_val,
				all_gt_related_x_placeholder_level5: all_gt_related_x_level5_val,
				all_gt_related_x_placeholder_level6: all_gt_related_x_level6_val,
				all_gt_related_y_placeholder_level2: all_gt_related_y_level2_val,
				all_gt_related_y_placeholder_level3: all_gt_related_y_level3_val,
				all_gt_related_y_placeholder_level4: all_gt_related_y_level4_val,
				all_gt_related_y_placeholder_level5: all_gt_related_y_level5_val,
				all_gt_related_y_placeholder_level6: all_gt_related_y_level6_val,
				all_gt_related_w_placeholder_level2: all_gt_related_w_level2_val,
				all_gt_related_w_placeholder_level3: all_gt_related_w_level3_val,
				all_gt_related_w_placeholder_level4: all_gt_related_w_level4_val,
				all_gt_related_w_placeholder_level5: all_gt_related_w_level5_val,
				all_gt_related_w_placeholder_level6: all_gt_related_w_level6_val,
				all_gt_related_h_placeholder_level2: all_gt_related_h_level2_val,
				all_gt_related_h_placeholder_level3: all_gt_related_h_level3_val,
				all_gt_related_h_placeholder_level4: all_gt_related_h_level4_val,
				all_gt_related_h_placeholder_level5: all_gt_related_h_level5_val,
				all_gt_related_h_placeholder_level6: all_gt_related_h_level6_val
				}
			LOSS, cls_LOSS_l2, cls_LOSS_l3, cls_LOSS_l4, cls_LOSS_l5, cls_LOSS_l6, reg_LOSS_l2, reg_LOSS_l3, reg_LOSS_l4, reg_LOSS_l5, reg_LOSS_l6, _ = sess.run([total_loss, classification_loss_level2, classification_loss_level3, classification_loss_level4, classification_loss_level5, classification_loss_level6,regression_loss_level2, regression_loss_level3, regression_loss_level4, regression_loss_level5, regression_loss_level6, train_op], feed_dict = feed_dict)
			LOSS_val, cls_LOSS_l2_val, cls_LOSS_l3_val, cls_LOSS_l4_val, cls_LOSS_l5_val, cls_LOSS_l6_val, reg_LOSS_l2_val, reg_LOSS_l3_val, reg_LOSS_l4_val, reg_LOSS_l5_val, reg_LOSS_l6_val, _ = sess.run([total_loss, classification_loss_level2, classification_loss_level3, classification_loss_level4, classification_loss_level5, classification_loss_level6,regression_loss_level2, regression_loss_level3, regression_loss_level4, regression_loss_level5, regression_loss_level6, train_op], feed_dict = feed_dict_val)
			with open(train_log, 'a') as log:
				log.write('Batch {}: Total_train_loss = {}, cls_train_loss_level2 = {}, cls_train_loss_level3 = {}, cls_train_loss_level4 = {}, cls_train_loss_level5 = {}, cls_train_loss_level6 = {}, reg_train_loss_level2 = {}, reg_train_loss_level3 = {}, reg_train_loss_level4 = {}, reg_train_loss_level5 = {}, reg_train_loss_level6 = {}'.format(step, LOSS, cls_LOSS_l2, cls_LOSS_l3, cls_LOSS_l4, cls_LOSS_l5, cls_LOSS_l6, reg_LOSS_l2, reg_LOSS_l3, reg_LOSS_l4, reg_LOSS_l5, reg_LOSS_l6))
				log.write('\n')
				log.write('          Total_val_loss = {}, cls_val_loss_level2 = {}, cls_val_loss_level3 = {}, cls_val_loss_level4 = {}, cls_val_loss_level5 = {}, cls_val_loss_level6 = {}, reg_val_loss_level2 = {}, reg_val_loss_level3 = {}, reg_val_loss_level4 = {}, reg_val_loss_level5 = {}, reg_val_loss_level6 = {}'.format(LOSS_val, cls_LOSS_l2_val, cls_LOSS_l3_val, cls_LOSS_l4_val, cls_LOSS_l5_val, cls_LOSS_l6_val, reg_LOSS_l2_val, reg_LOSS_l3_val, reg_LOSS_l4_val, reg_LOSS_l5_val, reg_LOSS_l6_val))
				log.write('\n')
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




















