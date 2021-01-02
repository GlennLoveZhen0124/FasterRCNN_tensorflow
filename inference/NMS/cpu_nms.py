import numpy as np
import json


def py_cpu_nms(y1, x1, y2, x2, scores, thresh):
	areas = (y2 - y1 + 1) * (x2 - x1 + 1)
	index = scores.argsort()[::-1]
	keep = []
	
	while index.size > 0:
		i = index[0]
		keep.append(i)
		y11 = np.maximum(y1[i], y1[index[1:]])
		x11 = np.maximum(x1[i], x1[index[1:]])
		y22 = np.maximum(y2[i], y2[index[1:]])
		x22 = np.maximum(x2[i], x2[index[1:]])
		
		w = np.maximum(0, x22 - x11 + 1)
		h = np.maximum(0, y22 - y11 + 1)
		
		overlaps = w * h
		ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
		
		idx = np.where(ious < thresh)[0]
		index = index[idx + 1]
	return keep

