import tensorflow as tf
import numpy as np
import math
import sys
import cv2
import os
from tqdm import tqdm


class Data(object):	
	def __init__(self, dataset_root):
		self.dataset_root = dataset_root
		self.anchor_ratios = [1., 0.5, 2.]
		self.anchor_scales = dict(level2 = [512], level3 = [256], level4 = [128], level5 = [64], level6 = [32])
		self.strides = dict(level2 = 4, level3 = 8, level4 = 16, level5 = 32, level6 = 64)
		#self.data = self.collect_data()
		self.data, self.img_path = self.collect_data()
		self.num = len(self.data)
		self.start = 0
		self.end = 0
	
	def Hprint(self, content):
		print('\033[1;36;40m')
		print(content)
		print('\033[0m')
	
	def collect_data(self):
		data = []
		img_path = []
		files = os.listdir(self.dataset_root)
		self.Hprint('Collecting data ... ...')
		for i in tqdm(range(len(files))):
			f = files[i]
			f_path = os.path.join(self.dataset_root, f)
			p, ext = os.path.splitext(f_path)
			if ext != '.xml':
				image_path = os.path.join(self.dataset_root, f)
				img = cv2.imread(image_path)
				if img.shape != (640,640,3):
					cv2.resize(img, (640,640))
				data.append(img)
				img_path.append(image_path)
		return data, img_path
	
	def generate_level_anchors(self, img_h, img_w, level):
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
		for (w,h) in sizes:
			all_anchors_x.append(x_coords)
			all_anchors_y.append(y_coords)
			all_anchors_w.append(np.ones(x_coords.shape) * w)
			all_anchors_h.append(np.ones(x_coords.shape) * h)
		all_anchors_x = np.stack(all_anchors_x, axis=-1)
		all_anchors_y = np.stack(all_anchors_y, axis=-1)
		all_anchors_w = np.stack(all_anchors_w, axis=-1)
		all_anchors_h = np.stack(all_anchors_h, axis=-1)
		return all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h
	
	def get_batch(self, batch_size = 1):
		assert batch_size == 1
		while True:
			if self.start + batch_size <= self.num:
				self.end = self.start + batch_size
				batch = self.data[self.start:self.end]
				image_path = self.img_path[self.start:self.end]
				img = batch[0] / 255.
				img = img.flatten()
				img = img[np.newaxis, :]
				img_h, img_w, _ = batch[0].shape
				ret_level2 = self.generate_level_anchors(img_h, img_w, level=2)
				ret_level3 = self.generate_level_anchors(img_h, img_w, level=3)
				ret_level4 = self.generate_level_anchors(img_h, img_w, level=4)
				ret_level5 = self.generate_level_anchors(img_h, img_w, level=5)
				ret_level6 = self.generate_level_anchors(img_h, img_w, level=6)
				self.start += batch_size
			else:
				self.start = 0
				self.end = batch_size
				batch = self.data[self.start:self.end]
				image_path = self.img_path[self.start:self.end]
				img = batch[0] / 255.
				img = img.flatten()
				img = img[np.newaxis, :]
				img_h, img_w, _ = batch[0].shape
				ret_level2 = self.generate_level_anchors(img_h, img_w, level=2)
				ret_level3 = self.generate_level_anchors(img_h, img_w, level=3)
				ret_level4 = self.generate_level_anchors(img_h, img_w, level=4)
				ret_level5 = self.generate_level_anchors(img_h, img_w, level=5)
				ret_level6 = self.generate_level_anchors(img_h, img_w, level=6)
				self.start += batch_size
			yield image_path[0], img, ret_level2, ret_level3, ret_level4, ret_level5, ret_level6


if __name__ == '__main__':
	data = Data('/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/dataset/val/image_xml_resize/')
	gen = data.get_batch(1)
	
	image_path, img, ret_level2, ret_level3, ret_level4, ret_level5, ret_level6 = gen.__next__()
	all_anchors_x_level2, all_anchors_y_level2, all_anchors_w_level2, all_anchors_h_level2 = ret_level2
	all_anchors_x_level3, all_anchors_y_level3, all_anchors_w_level3, all_anchors_h_level3 = ret_level3
	all_anchors_x_level4, all_anchors_y_level4, all_anchors_w_level4, all_anchors_h_level4 = ret_level4
	all_anchors_x_level5, all_anchors_y_level5, all_anchors_w_level5, all_anchors_h_level5 = ret_level5
	all_anchors_x_level6, all_anchors_y_level6, all_anchors_w_level6, all_anchors_h_level6 = ret_level6
	print('img: ', img.shape)
	print('all_anchors_x_level2: ', all_anchors_x_level2.shape)
	print('all_anchors_y_level2: ', all_anchors_y_level2.shape)
	print('all_anchors_w_level2: ', all_anchors_w_level2.shape)
	print('all_anchors_h_level2: ', all_anchors_h_level2.shape)
	print('all_anchors_x_level3: ', all_anchors_x_level3.shape)
	print('all_anchors_y_level3: ', all_anchors_y_level3.shape)
	print('all_anchors_w_level3: ', all_anchors_w_level3.shape)
	print('all_anchors_h_level3: ', all_anchors_h_level3.shape)
	print('all_anchors_x_level4: ', all_anchors_x_level4.shape)
	print('all_anchors_y_level4: ', all_anchors_y_level4.shape)
	print('all_anchors_w_level4: ', all_anchors_w_level4.shape)
	print('all_anchors_h_level4: ', all_anchors_h_level4.shape)
	print('all_anchors_x_level5: ', all_anchors_x_level5.shape)
	print('all_anchors_y_level5: ', all_anchors_y_level5.shape)
	print('all_anchors_w_level5: ', all_anchors_w_level5.shape)
	print('all_anchors_h_level5: ', all_anchors_h_level5.shape)
	print('all_anchors_x_level6: ', all_anchors_x_level6.shape)
	print('all_anchors_y_level6: ', all_anchors_y_level6.shape)
	print('all_anchors_w_level6: ', all_anchors_w_level6.shape)
	print('all_anchors_h_level6: ', all_anchors_h_level6.shape)









