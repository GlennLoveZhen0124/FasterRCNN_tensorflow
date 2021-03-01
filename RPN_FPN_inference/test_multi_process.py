import os
import cv2
import sys
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Manager, Process


class Data(object):
	def __init__(self, dataset_root):
		self.dataset_root = dataset_root
		self.files = os.listdir(self.dataset_root)
		self.files = [f for f in self.files if '.jpg' in f]
		self.files = [os.path.join(self.dataset_root,f) for f in self.files]
		self.data = Manager().list(self.files)
		self.num = len(self.data)
	
	def read_image(self, start, end):
		'''
		image_path = self.data[index]
		img = cv2.imread(image_path)
		self.data[index] = img
		'''
		for i in tqdm(range(start,end)):
			image_path = self.data[i]
			img = cv2.imread(image_path)
			self.data[i] = img
	
	def test(self, start, end):
		for i in tqdm(range(start,end)):
			time.sleep(0.005)
			pass
	
	def collect_data(self, workers = 1):
		interval = self.num//workers
		starts, ends = [], []
		for i in range(workers):
			starts.append(i*interval)
			ends.append((i+1)*interval)
		ends[-1] = self.num
		processes = []
		for i in range(workers):
			p = Process(target = self.read_image, args = (starts[i], ends[i]))
			p.start()
			processes.append(p)
		for p in processes:
			p.join()
		
		'''
		start1,end1 = 0, self.num//2
		start2,end2 = self.num//2, self.num
		p1 = Process(target = self.read_image, args = (start1,end1))
		p2 = Process(target = self.read_image, args = (start2,end2))
		p1.start()
		p2.start()
		p1.join()
		p2.join()
		'''


if __name__ == '__main__':
	folder = sys.argv[1]
	data = Data(folder)
	print(data.data[0])
	print(type(data.data), len(data.data))
	print(type(data.data[0]))
	data.collect_data()
	print(type(data.data[0]))




