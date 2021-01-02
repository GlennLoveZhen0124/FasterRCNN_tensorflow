import tensorflow as tf
import shutil
import os


class Utils(object):
	def __init__(self):
		pass

	def coords_convertion(self, labels, net_reg_x, net_reg_y, net_reg_w, net_reg_h, gt_related_x, gt_related_y, gt_related_w, gt_related_h, anchors_x, anchors_y, anchors_w, anchors_h, epsilon = 0.0000000000001):
		net_reg_x = tf.cast(net_reg_x, tf.float64)
		net_reg_y = tf.cast(net_reg_y, tf.float64)
		net_reg_w = tf.cast(net_reg_w, tf.float64)
		net_reg_h = tf.cast(net_reg_h, tf.float64)
		ind = tf.where(tf.not_equal(labels, -1))
		net_reg_x = tf.squeeze(net_reg_x)
		net_reg_y = tf.squeeze(net_reg_y)
		net_reg_w = tf.squeeze(net_reg_w)
		net_reg_h = tf.squeeze(net_reg_h)
		batch_labels = tf.gather_nd(labels, ind)
		batch_net_reg_x = tf.gather_nd(net_reg_x, ind)
		batch_net_reg_y = tf.gather_nd(net_reg_y, ind)
		batch_net_reg_w = tf.gather_nd(net_reg_w, ind)
		batch_net_reg_h = tf.gather_nd(net_reg_h, ind)
		batch_gt_related_x = tf.gather_nd(gt_related_x, ind)
		batch_gt_related_y = tf.gather_nd(gt_related_y, ind)
		batch_gt_related_w = tf.gather_nd(gt_related_w, ind)
		batch_gt_related_h = tf.gather_nd(gt_related_h, ind)
		batch_anchors_x = tf.gather_nd(anchors_x, ind)
		batch_anchors_y = tf.gather_nd(anchors_y, ind)
		batch_anchors_w = tf.gather_nd(anchors_w, ind)
		batch_anchors_h = tf.gather_nd(anchors_h, ind)
		batch_tx = (tf.cast(batch_net_reg_x, tf.float32) - tf.cast(batch_anchors_x, tf.float32)) / batch_anchors_w
		batch_ty = (tf.cast(batch_net_reg_y, tf.float32) - tf.cast(batch_anchors_y, tf.float32)) / batch_anchors_h
		batch_tw = tf.log(tf.cast(batch_net_reg_w, tf.float32) / batch_anchors_w + epsilon)
		batch_th = tf.log(tf.cast(batch_net_reg_h, tf.float32) / batch_anchors_h + epsilon)
		batch_tx_star = (batch_gt_related_x - batch_anchors_x) / batch_anchors_w
		batch_ty_star = (batch_gt_related_y - batch_anchors_y) / batch_anchors_h
		batch_tw_star = tf.log(batch_gt_related_w / batch_anchors_w + epsilon)
		batch_th_star = tf.log(batch_gt_related_h / batch_anchors_h + epsilon)
		return batch_tx, batch_ty, batch_tw, batch_th, batch_tx_star, batch_ty_star, batch_tw_star, batch_th_star
	
	def copy_files(self, orig_key, dst_key, folder):
		for f in os.listdir(folder):
			if orig_key in f:
				new_f = f.replace(orig_key, dst_key)
				orig = os.path.join(folder, f)
				dst = os.path.join(folder, new_f)
				shutil.copy2(orig, dst)
	









