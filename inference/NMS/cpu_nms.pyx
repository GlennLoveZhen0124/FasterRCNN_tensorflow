import numpy as np
cimport numpy as np


cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
	return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
	return a if a <= b else b

def py_cpu_nms(np.ndarray[np.float32_t, ndim = 1] y1, np.ndarray[np.float32_t, ndim = 1] x1, np.ndarray[np.float32_t, ndim = 1] y2, np.ndarray[np.float32_t, ndim = 1] x2, np.ndarray[np.float32_t, ndim = 1] scores, np.float thresh):
	cdef np.ndarray[np.float32_t, ndim = 1] areas = (y2 - y1 + 1) * (x2 - x1 + 1)
	cdef np.ndarray[np.int_t, ndim = 1] index = scores.argsort()[::-1]
	keep = []
	cdef int bboxes_num = scores.shape[0]
	cdef np.ndarray[np.int_t, ndim = 1] suppressed = np.zeros(bboxes_num, np.int)
	
	cdef int i, j, i_, j_
	
	cdef np.float32_t x1_i, y1_i, x2_i, y2_i, area_i
	cdef np.float32_t w, h
	cdef np.float32_t overlap, iou
	
	for i_ in range(bboxes_num):
		i = index[i_]
		if suppressed[i] == 1:
			continue
		keep.append(i)
		
		y1_i = y1[i]
		x1_i = x1[i]
		y2_i = y2[i]
		x2_i = x2[i]
		area_i = areas[i]
		for j_ in range(i_+1, bboxes_num):
			j = index[j_]
			if suppressed[j] == 1:
				continue
			yy1 = max(y1_i, y1[j])
			xx1 = max(x1_i, x1[j])
			yy2 = min(y2_i, y2[j])
			xx2 = min(x2_i, x2[j])
			
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			overlap = w * h
			iou = overlap / (area_i + areas[j] - overlap)
			if iou > thresh:
				suppressed[j] = 1
	
	return keep

