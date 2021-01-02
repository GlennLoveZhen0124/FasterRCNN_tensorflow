import tensorflow as tf
import numpy as np
import scipy.misc
import random
import math
import json
import sys
import cv2
import os
from PIL import Image, ImageDraw


class Data(object):
	def __init__(self, dataset_root, proposal_root, COCO_name_file):
		self.dataset_root = dataset_root
		self.proposal_root = proposal_root
		self.COCO_names = self.get_COCO_names(COCO_name_file)
		self.anchor_scales = [64, 128, 256, 512]
		self.anchor_ratios = [1., 0.5, 2.]
		#self.sizes = [(s,int(s*r)) for s in self.anchor_scales for r in self.anchor_ratios]
		self.sizes = [(int(s * math.sqrt(r)), int(s * math.sqrt(1/r))) for s in self.anchor_scales for r in self.anchor_ratios]
		self.data, self.img_path, self.xml_path, self.proposal_path = self.collect_data()
		self.num = len(self.data)
		self.start = 0
		self.end = 0
	
	def IOU(self, box1, box2):
		#left1, top1, right1, bottom1 = box1
		#left2, top2, right2, bottom2 = box2
		top1, left1, bottom1, right1 = box1
		top2, left2, bottom2, right2 = box2
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
	
	def get_COCO_names(self, COCO_name_file):
		with open(COCO_name_file) as f:
			data = f.readline().strip()
		jsondata = json.loads(data)
		return jsondata
	
	def collect_data(self):
		data = []
		img_path = []
		xml_paths = []
		proposal_paths = []
		ind = 0
		for f in os.listdir(self.dataset_root):
			name, ext = f.split('.')
			if ext == 'xml':
				xml_path = os.path.join(self.dataset_root, f)
				image_path = os.path.join(self.dataset_root, name+'.jpg')
				proposal_path = os.path.join(self.proposal_root, name + '.txt')
				img = cv2.imread(image_path)
				if img.shape != (640,640,3):
					img = cv2.resize(img, (640,640))
				'''
				img = img / 255.
				img = img.flatten()
				img = img[np.newaxis, :]
				'''
				data.append(img)
				img_path.append(image_path)
				xml_paths.append(xml_path)
				proposal_paths.append(proposal_path)
				print(ind)
				ind += 1
		return data, img_path, xml_paths, proposal_paths
	
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
	
	def get_bboxes(self, img, xml_path):
		img_h, img_w, img_c = img.shape
		bboxes = []
		with open(xml_path) as f:
			line = f.readline().strip()
			while line:
				if '</name>' in line:
					cls = line[6:-7]
				elif '</xmin>' in line:
					left = int(float(line[6:-7]))
					if left < 0:
						left = 0
				elif '</ymin>' in line:
					top = int(float(line[6:-7]))
					if top < 0:
						top = 0
				elif '</xmax>' in line:
					right = int(float(line[6:-7]))
					if right > img_w:
						right = img_w
				elif '</ymax>' in line:
					bottom = int(float(line[6:-7]))
					if bottom > img_h:
						bottom = img_h
					x,y,w,h = (left + right) // 2, (top + bottom) // 2, right-left, bottom-top
					bboxes.append([cls, x, y, w, h])
				line = f.readline().strip()
		return bboxes
	
	def get_proposals(self, proposal_path):
		proposals = []
		with open(proposal_path) as f:
			line = f.readline().strip()
			while line:
				y1, x1, y2, x2 = line.split(',')
				y1, x1, y2, x2 = float(y1), float(x1), float(y2), float(x2)
				proposals.append([y1,x1,y2,x2])
				line = f.readline().strip()
		proposals = np.array(proposals)
		return proposals
	
	def get_batch(self, batch_size = 64):
		while True:
			if self.start + 1 <= self.num:
				self.end = self.start + 1
				batch = self.data[self.start:self.end]
				batch_xml = self.xml_path[self.start:self.end]
				batch_proposal = self.proposal_path[self.start:self.end]
				img = batch[0] / 255.
				img = img.flatten()
				img = img[np.newaxis, :]
				all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = self.generate_anchors(batch[0])
				bboxes = self.get_bboxes(batch[0], batch_xml[0])
				self.start += 1
			else:
				self.start = 0
				self.end = 1
				batch = self.data[self.start:self.end]
				batch_xml = self.xml_path[self.start:self.end]
				batch_proposal = self.proposal_path[self.start:self.end]
				img = batch[0] / 255.
				img = img.flatten()
				img = img[np.newaxis, :]
				all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h = self.generate_anchors(batch[0])
				bboxes = self.get_bboxes(batch[0], batch_xml[0])
				self.start += 1
			all_GT_cls = [self.COCO_names[x[0]] for x in bboxes]
			all_GT_x = [x[1] for x in bboxes]
			all_GT_y = [x[2] for x in bboxes]
			all_GT_w = [x[3] for x in bboxes]
			all_GT_h = [x[4] for x in bboxes]
			all_GT_cls = np.array(all_GT_cls)
			all_GT_x = np.array(all_GT_x)
			all_GT_y = np.array(all_GT_y)
			all_GT_w = np.array(all_GT_w)
			all_GT_h = np.array(all_GT_h)
			all_GT_y1 = all_GT_y - all_GT_h/2
			all_GT_x1 = all_GT_x - all_GT_w/2
			all_GT_y2 = all_GT_y + all_GT_h/2
			all_GT_x2 = all_GT_x + all_GT_w/2
			all_GT_boxes = np.vstack([all_GT_y1, all_GT_x1, all_GT_y2, all_GT_x2])	# shape is (4, 3), 3 boxes, 4 coords each
			#all_GT_boxes = all_GT_boxes.T
			#print('all_GT_boxes: ', all_GT_boxes, all_GT_boxes.shape)
			proposals = self.get_proposals(batch_proposal[0])	# [y1,x1,y2,x2] i.e. [top, left, bottom, right]
			unsort_P = []
			pos_num = 0
			for proposal in proposals:
				iou = self.IOU(proposal, all_GT_boxes)		# shape is (3, ), i.e. number of GT boxes
				#print('iou: ', iou, iou.shape)
				iou_max_index = np.argmax(iou)
				box_max = all_GT_boxes[:, iou_max_index]	# shape is (4, ), i.e. [y1, x1, y2, x2]
				box_max_y1, box_max_x1, box_max_y2, box_max_x2 = box_max
				#print('box_max: ', box_max, box_max.shape)
				max_iou = np.max(iou)
				cls_max = all_GT_cls[iou_max_index]
				if max_iou < 0.5:
					cls_max = 0
				else:
					pos_num += 1
				#print('IOU: ', iou, iou_max_index, cls_max)
				unsort_P.append([max_iou, cls_max, proposal, box_max_y1, box_max_x1, box_max_y2, box_max_x2])
			sort_P = sorted(unsort_P, key = lambda x:x[0])[::-1]
			#print('sort_P: ', sort_P)
			batch_data_use = sort_P[:batch_size]
			random.shuffle(batch_data_use)
			proposals_in_batch = [x[2] for x in batch_data_use]
			proposals_in_batch = np.vstack(proposals_in_batch)
			classes_in_batch = [x[1] for x in batch_data_use]
			classes_in_batch_onehot = np.eye(81)[classes_in_batch]
			box_max_y1_in_batch = [x[3] for x in batch_data_use]
			box_max_y1_in_batch = np.array(box_max_y1_in_batch)	# shape is (64, ), i.e. (batch_size, )
			box_max_x1_in_batch = [x[4] for x in batch_data_use]
			box_max_x1_in_batch = np.array(box_max_x1_in_batch)	# shape is (64, ), i.e. (batch_size, )
			box_max_y2_in_batch = [x[5] for x in batch_data_use]
			box_max_y2_in_batch = np.array(box_max_y2_in_batch)	# shape is (64, ), i.e. (batch_size, )
			box_max_x2_in_batch = [x[6] for x in batch_data_use]
			box_max_x2_in_batch = np.array(box_max_x2_in_batch)	# shape is (64, ), i.e. (batch_size, )
			#print('proposals_in_batch: ', proposals_in_batch, proposals_in_batch.shape, type(proposals_in_batch[0]))
			#print('classes_in_batch_onehot: ', classes_in_batch_onehot, classes_in_batch_onehot.shape)
			#print('box_max_y1_in_batch: ', box_max_y1_in_batch, type(box_max_y1_in_batch), box_max_y1_in_batch.shape)
			yield batch_xml, img, proposals_in_batch, classes_in_batch_onehot, box_max_y1_in_batch, box_max_x1_in_batch, box_max_y2_in_batch, box_max_x2_in_batch, pos_num
			#yield batch_xml, img, proposals , all_anchors_x, all_anchors_y, all_anchors_w, all_anchors_h, all_GT_cls, all_GT_x, all_GT_y, all_GT_w, all_GT_h,all_GT_boxes


# tmp draw func 
def draw_rect(img, boxes, save_path):
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
	pil_img.save(save_path)


if __name__ == '__main__':
	
	save_folder = '/root/Faster_RCNN_tensorflow/dataset/debug_draw/'
	data = Data('/root/Faster_RCNN_tensorflow/dataset/test_for_resize/', '/root/Faster_RCNN_tensorflow/dataset/trainval_resize_proposals_VGGpretrain_step2/', '/root/Faster_RCNN_tensorflow/COCO_names.names')
	gen = data.get_batch(32)
	
	
	for i in range(len(os.listdir('/root/Faster_RCNN_tensorflow/dataset/trainval_resize_proposals_VGGpretrain_step2/'))):
		batch_xml, img, proposals_in_batch, classes_in_batch_onehot, box_max_y1_in_batch, box_max_x1_in_batch, box_max_y2_in_batch, box_max_x2_in_batch, pos_num = gen.__next__()
		
		print('box_max_y1_in_batch: ', box_max_y1_in_batch, box_max_y1_in_batch.shape)
		print('box_max_x1_in_batch: ', box_max_x1_in_batch, box_max_x1_in_batch.shape)
		print('proposals: ', proposals_in_batch, proposals_in_batch.shape)
		
		
		'''
		print('positive number: ', pos_num)
		print('batch_xml: ', batch_xml)
		print('proposals: ', proposals_in_batch, proposals_in_batch.shape)
		xml_path = batch_xml[0]
		basename = os.path.basename(xml_path)
		name, _ = basename.split('.')
		image_path = os.path.join('/root/Faster_RCNN_tensorflow/dataset/test_for_resize/', name+'.jpg')
		save_path = os.path.join(save_folder, name+'.jpg')
		print('SAVE PATH: ', save_path)
		img = scipy.misc.imread(image_path)
		draw_rect(img, proposals_in_batch, save_path)
		'''
		
		'''
		with open('positive_number.log', 'a') as log:
			log.write(str(pos_num))
			log.write('\n')
		'''
		break
	
	
	














