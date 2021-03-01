import os
import sys
import cv2
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Data(object):
	def __init__(self, dataset_root):
		self.dataset_root = dataset_root
		self.anchor_ratios = [1., 0.5, 2.]
		self.anchor_scales = dict(level2 = [512], level3 = [256], level4 = [128], level5 = [64], level6 = [32])
		self.strides = dict(level2 = 4, level3 = 8, level4 = 16, level5 = 32, level6 = 64)
		self.data = self.collect_data()
		self.data_num = len(self.data)
		self.cur = 0
	
	def Hprint(self, content):
		print('\033[1;36;40m')
		print(content)
		print('\033[0m')
	
	def IOU(self, box1, box2):
		left1, top1, right1, bottom1 = box1
		left2, top2, right2, bottom2 = box2
		area1 = (right1 - left1) * (bottom1 - top1)
		area2 = (right2 - left2) * (bottom2 - top2)
		left = np.maximum(left1, left2)
		right = np.minimum(right1, right2)
		top = np.maximum(top1, top2)
		bottom = np.minimum(bottom1, bottom2)
		intersection = np.maximum(0, (right - left)) * np.maximum(0, (bottom - top))
		union = area1 + area2 - intersection
		iou = intersection.astype(np.float32) / union
		return iou
		
	def collect_data(self):
		data = []
		files = os.listdir(self.dataset_root)
		self.Hprint('Collecting data ... ...')
		for i in tqdm(range(len(files))):
			f = files[i]
			f_path = os.path.join(self.dataset_root, f)
			p, ext = os.path.splitext(f_path)
			if ext != '.xml':
				image_path = os.path.join(self.dataset_root, f)
				xml_path = p + '.xml'
				data.append((xml_path, image_path))
		return data
	
	def read_xml(self, xml_file, image_file):
		locations = []
		img = cv2.imread(image_file)
		img_h, img_w, img_c = img.shape
		with open(xml_file) as f:
			line = f.readline().strip()
			while line:
				if '</xmin>' in line:
					left = int(float(line[6:-7]))
					if left < 0:
						left = 0
				if '</ymin>' in line:
					top = int(float(line[6:-7]))
					if top < 0:
						top = 0
				if '</xmax>' in line:
					right = int(float(line[6:-7]))
					if top > img_w:
						right = img_w
				if '</ymax>' in line:
					bottom = int(float(line[6:-7]))
					if bottom > img_h:
						bottom = img_h
					locations.append([left,top,right,bottom])
				line = f.readline().strip()
		locations = np.array(locations)
		return locations, img
	
	def resize_image_and_coords(self, img, locations):
		img_h, img_w, img_c = img.shape
		if img_h > img_w:
			w_new = 640.
			ratio = w_new / img_w
			h_new = img_h * ratio
			img = cv2.resize(img, (int(w_new),int(h_new)))
		else:
			h_new = 640.
			ratio = h_new / img_h
			w_new = img_w * ratio
			img = cv2.resize(img, (int(w_new),int(h_new)))
		return ratio, img
	
	def get_level_anchors(self, img_h, img_w, ground_truth, max_positive_anchor_nums, batch_size, level):
		anchor_scales = self.anchor_scales['level{}'.format(level)]
		sizes = [(int(s * math.sqrt(r)), int(s * math.sqrt(1/r))) for s in anchor_scales for r in self.anchor_ratios]
		stride = self.strides['level{}'.format(level)]
		img_h_rescale, img_w_rescale = img_h//stride, img_w//stride
		x_coords = np.ones([img_h_rescale,img_w_rescale])
		y_coords = np.ones([img_h_rescale,img_w_rescale])
		if img_w_rescale < len(range(0, img_w, stride)):
			x_coords = x_coords * np.array(list(range(0, img_w, stride))[:-1]) + stride//2
		else:
			x_coords = x_coords * np.array(list(range(0, img_w, stride))) + stride//2
		if img_h_rescale < len(range(0, img_h, stride)):
			y_coords = y_coords.T * np.array(list(range(0, img_h, stride))[:-1]) + stride//2
			y_coords = y_coords.T
		else:
			y_coords = y_coords.T * np.array(list(range(0, img_h, stride))) + stride//2
			y_coords = y_coords.T
		all_anchors_x = []
		all_anchors_y = []
		all_anchors_w = []
		all_anchors_h = []
		all_labels = []
		all_gt_related = []
		for (w, h) in sizes:
			labels = np.zeros(x_coords.shape)
			gt_related = np.ones(labels.shape) * (-1)
			anchor_left = x_coords - w//2
			anchor_right = x_coords + w//2
			anchor_top = y_coords - h//2
			anchor_bottom = y_coords + h//2
			border_index_left = np.where(anchor_left >= 0)
			border_index_right = np.where(anchor_right < img_w)
			border_index_top = np.where(anchor_top >= 0)
			border_index_bottom = np.where(anchor_bottom < img_h)
			labels[border_index_left] = -1
			labels[border_index_right] = -1
			labels[border_index_top] = -1
			labels[border_index_bottom] = -1
			for gt_ind in range(ground_truth.shape[0]):
				gt_box = ground_truth[gt_ind]
				iou = self.IOU([anchor_left, anchor_top, anchor_right, anchor_bottom], gt_box)
				label_index_07 = np.where(iou >= 0.7)
				labels[label_index_07] = 1
				gt_related[label_index_07] = gt_ind
				if label_index_07[0].shape == (0,) and label_index_07[1].shape == (0,):		# max iou is smaller than 0.7
					label_index_max = np.unravel_index(iou.argmax(), iou.shape)
					if iou[label_index_max] > 0.3:
						labels[label_index_max] = 1
						gt_index_cur = 0
						iou_max_cur = -1
						for tmp_gt_index in range(ground_truth.shape[0]):
							iou_cur = self.IOU([anchor_left[label_index_max], anchor_top[label_index_max], anchor_right[label_index_max], anchor_bottom[label_index_max]], ground_truth[tmp_gt_index])
							if iou_cur > iou_max_cur:
								iou_max_cur = iou_cur
								gt_index_cur = tmp_gt_index
							else:
								pass
						gt_related[label_index_max] = gt_index_cur
				neg_cond1 = iou <= 0.3
				neg_cond2 = labels != 1							# labels != 1 is for preventing rewrite
				neg_cond = neg_cond1 * neg_cond2
				label_index_03 = np.where(neg_cond)
				labels[label_index_03] = 0
				
			all_anchors_x.append(x_coords)
			all_anchors_y.append(y_coords)
			all_anchors_w.append(np.ones(x_coords.shape) * w)
			all_anchors_h.append(np.ones(x_coords.shape) * h)
			all_labels.append(labels)
			all_gt_related.append(gt_related)
		all_anchors_x = np.stack(all_anchors_x, axis=-1)
		all_anchors_y = np.stack(all_anchors_y, axis=-1)
		all_anchors_w = np.stack(all_anchors_w, axis=-1)
		all_anchors_h = np.stack(all_anchors_h, axis=-1)
		all_labels = np.stack(all_labels, axis=-1)
		all_gt_related = np.stack(all_gt_related, axis=-1)
		all_gt_related = all_gt_related.astype(np.int32)
		all_gt_related_x = ground_truth[:,0] + (ground_truth[:,2] - ground_truth[:,0]) / 2
		all_gt_related_y = ground_truth[:,1] + (ground_truth[:,3] - ground_truth[:,1]) / 2
		all_gt_related_w = ground_truth[:,2] - ground_truth[:,0]
		all_gt_related_h = ground_truth[:,3] - ground_truth[:,1]
		
		# find all batch_size//2 positive and all batch_size//2 negative
		positive_inds = np.where(all_labels == 1)
		negative_inds = np.where(all_labels == 0)
		all_positive_labels = all_labels[positive_inds]
		all_negative_labels = all_labels[negative_inds]
		if all_positive_labels.shape[0] > max_positive_anchor_nums:		# batch_size = 256, pos : neg <= 1 : 1
			positive_anchor_nums = max_positive_anchor_nums
			disable_pos_inds = np.random.choice(list(range(positive_inds[0].shape[0])), all_positive_labels.shape[0] - max_positive_anchor_nums, replace=False)
			indexes_pos = []
			for i in range(len(positive_inds)):
				indexes_pos.append(positive_inds[i][disable_pos_inds])
			for i in range(indexes_pos[0].shape[0]):
				all_labels[indexes_pos[0][i]][indexes_pos[1][i]][indexes_pos[2][i]] = -1
		else:
			positive_anchor_nums = all_positive_labels.shape[0]
		
		if all_negative_labels.shape[0] > batch_size - positive_anchor_nums:
			disable_neg_inds = np.random.choice(list(range(negative_inds[0].shape[0])), all_negative_labels.shape[0] - (batch_size - positive_anchor_nums), replace=False)
			indexes_neg = []
			for i in range(len(negative_inds)):
				indexes_neg.append(negative_inds[i][disable_neg_inds])
			for i in range(indexes_neg[0].shape[0]):
				all_labels[indexes_neg[0][i]][indexes_neg[1][i]][indexes_neg[2][i]] = -1
		else:
			pass		# there's no possibility that all_negative_labels.shape[0] <= batch_size - positive_anchor_nums
		
		all_gt_related_x = all_gt_related_x[all_gt_related]
		all_gt_related_y = all_gt_related_y[all_gt_related]
		all_gt_related_w = all_gt_related_w[all_gt_related]
		all_gt_related_h = all_gt_related_h[all_gt_related]
		
		return all_labels, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h, all_gt_related_x, all_gt_related_y, all_gt_related_w, all_gt_related_h
	
	def get_batch_anchors(self, batch_size):
		'''
		batch_size: 256
		ret:
			all_labels_level2		--->	np array with shape (w,h,3), same as the feature map, with entry value in [-1, 0, 1], -1 is deprecated, 0 is background, 1 is foreground
			all_anchors_x_level2		--->	np array with shape (w,h,3), same as the feature map, with each anchor_x value in entry
			all_anchors_y_level2		--->	np array with shape (w,h,3), same as the feature map, with each anchor_y value in entry
			all_anchors_w_level2		--->	np array with shape (w,h,3), same as the feature map, with each anchor_w value in entry
			all_anchors_h_level2		--->	np array with shape (w,h,3), same as the feature map, with each anchor_h value in entry
			all_gt_related_x_level2	--->	np array with shape (w,h,3), same as the feature map, with each max_iou_gt_x value in entry
			all_gt_related_y_level2	--->	np array with shape (w,h,3), same as the feature map, with each max_iou_gt_y value in entry
			all_gt_related_w_level2	--->	np array with shape (w,h,3), same as the feature map, with each max_iou_gt_w value in entry
			all_gt_related_h_level2	--->	np array with shape (w,h,3), same as the feature map, with each max_iou_gt_h value in entry
			level3 ...
			level4 ...
			level5 ...
			level6 ...
		'''
		max_positive_anchor_nums = batch_size // 2
		
		while True:
			if self.cur < self.data_num:
				data_cur = self.data[self.cur]
				self.cur += 1
			else:
				data_cur = self.data[0]
				self.cur = 1
			xml_path, image_path = data_cur
			locations, img = self.read_xml(xml_path, image_path)		# GT object locations and img
			ground_truth = locations
			img_h, img_w, img_c = img.shape
			ret_level2 = self.get_level_anchors(img_h, img_w, ground_truth, max_positive_anchor_nums, batch_size, level=2)
			ret_level3 = self.get_level_anchors(img_h, img_w, ground_truth, max_positive_anchor_nums, batch_size, level=3)
			ret_level4 = self.get_level_anchors(img_h, img_w, ground_truth, max_positive_anchor_nums, batch_size, level=4)
			ret_level5 = self.get_level_anchors(img_h, img_w, ground_truth, max_positive_anchor_nums, batch_size, level=5)
			ret_level6 = self.get_level_anchors(img_h, img_w, ground_truth, max_positive_anchor_nums, batch_size, level=6)
			img = img / 255.
			img = img.flatten()
			img = img[np.newaxis, :]
			yield image_path, img, ret_level2, ret_level3, ret_level4, ret_level5, ret_level6


if __name__ == '__main__':
	data = Data('/root/Faster_RCNN_tensorflow/dataset/test_for_resize/')
	
	'''
	gen = data.get_batch_anchors(256)
	all_anchors = gen.__next__()
	print('all_anchors: ', all_anchors, len(all_anchors), type(all_anchors[0]), type(all_anchors[0][0]), len(all_anchors[0][0]), type(all_anchors[0][0][0]), all_anchors[0][0][0].shape)
	test = np.dstack((all_anchors[0][0][0], all_anchors[1][0][0], all_anchors[3][0][0], all_anchors[4][0][0], all_anchors[5][0][0]))
	print(test.shape)
	'''
	
	#gen = data.get_batch_anchors(256)
	gen = data.get_batch_anchors(64)
	batch = gen.__next__()
	ret_level2 = batch[2]
	ret_level3 = batch[3]
	ret_level4 = batch[4]
	ret_level5 = batch[5]
	ret_level6 = batch[6]
	print('ret_level2: ', len(ret_level2), ret_level2[0].shape, ret_level2[1].shape, ret_level2[-1].shape)
	print('ret_level3: ', len(ret_level3), ret_level3[0].shape)
	print('ret_level4: ', len(ret_level4), ret_level4[0].shape)
	print('ret_level5: ', len(ret_level5), ret_level5[0].shape)
	print('ret_level6: ', len(ret_level6), ret_level6[0].shape)
	print(ret_level6[0])
	print(ret_level6[1])
	print(ret_level6[-1])
	
	x = ret_level6[0].flatten()
	count = 0
	for a in x:
		if a == 1:
			count += 1
	print(count)
	print(x)
	pos_level2 = np.where(ret_level2[0] == 1)
	print(pos_level2)
	
	'''
	print(batch[2].shape)
	print(0 in batch[2])
	print(data.sizes)
	print(batch[7], batch[7].shape)
	print('=======================')
	print(set(list(batch[7].flatten())))
	'''








