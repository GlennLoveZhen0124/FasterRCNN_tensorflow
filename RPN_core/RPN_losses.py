import tensorflow as tf


class Losses(object):
	def __init__(self):
		pass
	
	def smooth_L1(self, t, t_star):
		cond = tf.less(tf.abs(t - t_star), 1)
		smooth = tf.where(condition = cond, x = 0.5*tf.square(t - t_star), y = tf.abs(t - t_star) - 0.5)
		return smooth
	
	def cls_loss(self, labels, net_cls_0, net_cls_1):
		print('LABELS: ', labels)
		#print('NET_CLS: ', net_cls)
		#net_cls = tf.squeeze(input=net_cls.eval(), axis=0)
		#net_cls = net_cls[0][0]
		net_cls_0 = net_cls_0[0]
		net_cls_1 = net_cls_1[0]
		print('NET_CLS_0: ', net_cls_0)
		print('NET_CLS_1: ', net_cls_1)
		#print('NET_CLS: ', net_cls)
		ind = tf.where(tf.not_equal(labels, -1))
		batch_labels = tf.gather_nd(labels, ind)
		batch_labels = tf.cast(batch_labels, tf.int32)
		print('BATCH_LABELS: ', batch_labels)
		#batch_labels_testtest = tf.concat([batch_labels, batch_labels], axis=-1)
		#print('BATCH_LABELS: ', batch_labels_testtest)
		#batch_net_cls = tf.gather_nd(net_cls, ind)
		#print('batch_net_cls: ', batch_net_cls.get_shape().as_list())
		batch_net_cls_0 = tf.gather_nd(net_cls_0, ind)
		batch_net_cls_1 = tf.gather_nd(net_cls_1, ind)
		batch_net_cls = tf.stack([batch_net_cls_0, batch_net_cls_1], axis=-1)
		
		onehot_label = tf.one_hot(batch_labels, depth=2, on_value=0.95, off_value=0.05)
		print('ONEHOT_LABEL: ', onehot_label)
		classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = onehot_label, logits = batch_net_cls))
		return classification_loss
	
	def reg_loss(self, batch_tx, batch_ty, batch_tw, batch_th, batch_tx_star, batch_ty_star, batch_tw_star, batch_th_star):
		loss = self.smooth_L1(batch_tx, batch_tx_star) + self.smooth_L1(batch_ty, batch_ty_star) + self.smooth_L1(batch_tw, batch_tw_star) + self.smooth_L1(batch_th, batch_th_star)
		loss = tf.reduce_mean(loss)
		return loss


