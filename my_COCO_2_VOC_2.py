import os
import sys
import cv2
import numpy as np
import json

from PIL import Image, ImageDraw
import scipy.misc

#ann = './annotations/instances_val2014.json'
ann = sys.argv[1]
dst_xml_folder = sys.argv[2]
image_folder = sys.argv[3]
image_save_folder = sys.argv[4]
process_log = sys.argv[5]

with open (ann) as f:
	data = f.readline().strip()

D = json.loads(data)

images = D['images']
annotations = D['annotations']
categories = D['categories']


def get_image(image_id, images):
	for img in images:
		if img['id'] == image_id:
			return img['file_name']

def get_cls(category_id, categories):
	for cat in categories:
		if cat['id'] == category_id:
			return cat['name']

def get_bbox(id, annotations, categories, images):
	res = []
	for d in annotations:
		category_id = d['category_id']
		cls = get_cls(category_id, categories)
		if d['image_id'] == id:
			image_path = get_image(id, images)
			res.append([cls, d['bbox']])
	try:
		res.insert(0, image_path)
		return res
	except UnboundLocalError:
		return 'Not Exist.'

def resize_image(data, image_folder, image_save_folder):
	new_data = []
	image_path = data[0]
	img = cv2.imread(os.path.join(image_folder, image_path))
	img_h, img_w, img_c = img.shape
	if img_h == 640 and img_w == 640:
		#print('new_data: ', data)
		return data
	new_data.append(image_path)
	data = data[1:]
	if img_h == img_w:
		resized = cv2.resize(img, (640,640))
		ratio = 640 / float(img_h)
		for object in data:
			cls = object[0]
			coord = object[1]
			x, y, w, h = coord
			x_new, y_new, w_new, h_new = ratio*x, ratio*y, ratio*w, ratio*h
			object[1] = [x_new, y_new, w_new, h_new]
			new_data.append(object)
		#cv2.imwrite(os.path.join(image_folder, image_path), resized)
		#cv2.imwrite(os.path.join('./train2014_640/', image_path), resized)
		cv2.imwrite(os.path.join(image_save_folder, image_path), resized)
		#print('new_data: ', data)
		return new_data
	else:
		resized = cv2.resize(img, (640,640))
		ratio_h = 640 / float(img_h)
		ratio_w = 640 / float(img_w)
		cv2.imwrite(os.path.join(image_save_folder, image_path), resized)
		for object in data:
			cls = object[0]
			coord = object[1]
			x_obj, y_obj, w_obj, h_obj = coord
			x_obj_new, y_obj_new, w_obj_new, h_obj_new = int(x_obj * ratio_w), int(y_obj * ratio_h), int(w_obj * ratio_w), int(h_obj * ratio_h)
			object[1] = [x_obj_new, y_obj_new, w_obj_new, h_obj_new]
			new_data.append(object)
		return new_data

def create_xml(data, dst_xml_folder, image_folder, image_save_folder):
	data = resize_image(data, image_folder, image_save_folder)
	#print('new_data: ', data)s
	image_path = data[0]
	img = cv2.imread(os.path.join(image_folder, image_path))
	img_h, img_w, img_c = img.shape
	xml_name = image_path.split('.')[0]
	xml_path = xml_name + '.xml'
	xml_path = os.path.join(dst_xml_folder, xml_path)
	data = data[1:]
	with open(xml_path, 'a') as f:
		f.write('<annotation>')
		f.write('\n')
		f.write('\t<folder>xxx</folder>')
		f.write('\n')
		f.write('\t<filename>{}</filename>'.format(os.path.join(image_folder, image_path)))
		f.write('\n')
		f.write('<path>{}</path>'.format(image_path))
		f.write('\n')
		f.write('\t<source>')
		f.write('\n')
		f.write('\t\t<database>Unknown</database>')
		f.write('\n')
		f.write('\t</source>')
		f.write('\n')
		f.write('\t<size>')
		f.write('\n')
		f.write('\t\t<width>{}</width>'.format(img_w))
		f.write('\n')
		f.write('\t\t<height>{}</height>'.format(img_h))
		f.write('\n')
		f.write('\t\t<depth>{}</depth>'.format(img_c))
		f.write('\n')
		f.write('\t</size>')
		f.write('\n')
		f.write('\t<segmented>0</segmented>')
		f.write('\n')
		for object in data:
			cls = object[0]
			coord = object[1]
			x, y, w, h = coord
			'''
			xmin = int(x - w/2)
			xmax = int(x + w/2)
			ymin = int(y - h/2)
			ymax = int(y + h/2)
			'''
			xmin = x
			xmax = x + w
			ymin = y
			ymax = y + h
			f.write('\t<object>')
			f.write('\n')
			f.write('\t\t<name>{}</name>'.format(cls))
			f.write('\n')
			f.write('\t\t<pose>Unspecified</pose>')
			f.write('\n')
			f.write('\t\t<truncated>0</truncated>')
			f.write('\n')
			f.write('\t\t<difficult>0</difficult>')
			f.write('\n')
			f.write('\t\t<bndbox>')
			f.write('\n')
			f.write('\t\t\t<xmin>{}</xmin>'.format(xmin))
			f.write('\n')
			f.write('\t\t\t<ymin>{}</ymin>'.format(ymin))
			f.write('\n')
			f.write('\t\t\t<xmax>{}</xmax>'.format(xmax))
			f.write('\n')
			f.write('\t\t\t<ymax>{}</ymax>'.format(ymax))
			f.write('\n')
			f.write('\t\t</bndbox>')
			f.write('\n')
			f.write('\t</object>')
			f.write('\n')
		f.write('</annotation>')
	print('Done creating {}'.format(xml_path))


xml_orig_list = images.copy()
for xml_dict in xml_orig_list:
	id = xml_dict['id']
	#cls, bbox = get_bbox(id, annotations, categories, images)
	res = get_bbox(id, annotations, categories, images)
	#print(cls, bbox)
	print(res)
	#create_xml(data, dst_xml_folder, image_folder)
	image_path = os.path.join(image_folder, res[0])
	if not os.path.exists(image_path):
		print('Image not exists.')
		continue
	#with open('process.log', 'a') as log:
	#with open('process_val.log', 'a') as log:
	with open(process_log, 'a') as log:
		log.write(image_path)
		log.write('\n')
	create_xml(res, dst_xml_folder, image_folder, image_save_folder)
	#break
	


#print(annotations)
#print(categories, type(categories), len(categories))
#print(images)










