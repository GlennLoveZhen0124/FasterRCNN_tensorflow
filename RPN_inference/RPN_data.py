import tensorflow as tf
import numpy as np
import math
import sys
import cv2
import os


class Data(object):
	def __init__(self, dataset_root):
		self.dataset_root = dataset_root
		self.anchor_scales = [64, 128, 256, 512]
		self.anchor_ratios = [1., 0.5, 2.]
		#self.sizes = [(s,int(s*r)) for s in self.anchor_scales for r in self.anchor_ratios]
		self.sizes = [(int(s * math.sqrt(r)), int(s * math.sqrt(1/r))) for s in self.anchor_scales for r in self.anchor_ratios]
		self.data, self.img_path = self.collect_data()
		self.num = len(self.data)
		self.start = 0
		self.end = 0
		print(self.img_path[0])
		with open('img.log', 'w') as log:
			log.write(self.img_path[0])
	
	def collect_data(self):
		data = []
		img_path = []
		ind = 0
		for f in os.listdir(self.dataset_root):
			name, ext = f.split('.')
			if ext == 'xml':
				continue
			image_path = os.path.join(self.dataset_root, f)
			img = cv2.imread(image_path)
			if img.shape != (640,640,3):
				cv2.resize(img, (640,640))
			'''
			img = img / 255.
			img = img.flatten()
			img = img[np.newaxis, :]
			'''
			data.append(img)
			img_path.append(image_path)
			ind += 1
			print(ind)
		return data, img_path
	
	def generate_anchors(self, img):
		img_h, img_w, img_c = img.shape
		img_h_rescale, img_w_rescale = img_h//16, img_w//16
		x_coords = np.ones([img_h_rescale,img_w_rescale])
		y_coords = np.ones([img_h_rescale,img_w_rescale])
		if img_w_rescale < len(range(0, img_w, 16)):
			x_coords = x_coords * np.array(list(range(0, img_w, 16))[:-1]) + 8
		else:
			x_coords = x_coords * np.array(list(range(0, img_w, 16))) + 8
		if img_h_rescale < len(range(0, img_h, 16)):
			y_coords = y_coords.T * np.array(list(range(0, img_h, 16))[:-1]) + 8
			y_coords = y_coords.T
		else:
			y_coords = y_coords.T * np.array(list(range(0, img_h, 16))) + 8
			y_coords = y_coords.T
		all_anchors_x = []
		all_anchors_y = []
		all_anchors_w = []
		all_anchors_h = []
		for (w,h) in self.sizes:
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
				all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = self.generate_anchors(batch[0])
				self.start += batch_size
			else:
				self.start = 0
				self.end = batch_size
				batch = self.data[self.start:self.end]
				image_path = self.img_path[self.start:self.end]
				img = batch[0] / 255.
				img = img.flatten()
				img = img[np.newaxis, :]
				all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = self.generate_anchors(batch[0])
				self.start += batch_size
			yield image_path[0], img, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h


if __name__ == '__main__':
	data = Data('/notebooks/dataset/test/')
	gen = data.get_batch(1)
	#print(gen.__next__())
	image_path, img, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = gen.__next__()
	print('img: ', img)
	print('all_anchors_x: ', all_anchors_x)
	print('all_anchors_y: ', all_anchors_y)
	print('all_anchors_w: ', all_anchors_w)
	print('all_anchors_h: ', all_anchors_h)
	print('============================================')
	print('all_anchors_x: ', all_anchors_x.shape)
	print('all_anchors_y: ', all_anchors_y.shape)
	print('all_anchors_w: ', all_anchors_w.shape)
	print('all_anchors_h: ', all_anchors_h.shape)










