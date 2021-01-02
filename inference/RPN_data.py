import tensorflow as tf
import numpy as np
import math
import sys
import cv2
import os


class Data(object):
	def __init__(self):
		self.anchor_scales = [64, 128, 256, 512]
		self.anchor_ratios = [1., 0.5, 2.]
		#self.sizes = [(s,int(s*r)) for s in self.anchor_scales for r in self.anchor_ratios]
		self.sizes = [(int(s * math.sqrt(r)), int(s * math.sqrt(1/r))) for s in self.anchor_scales for r in self.anchor_ratios]
	
	def collect_data(self, image_path):
		img = cv2.imread(image_path)
		print('img.shape: ', img.shape)
		if img.shape != (640,640,3):
			img = cv2.resize(img, (640,640))
		print('img.shape: ', img.shape)
		return img
	
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
	
	def get_batch(self, image_path):
		img_ = self.collect_data(image_path)
		img = img_ / 255.
		img = img.flatten()
		img = img[np.newaxis, :]
		all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = self.generate_anchors(img_)
		return img, all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h
	










