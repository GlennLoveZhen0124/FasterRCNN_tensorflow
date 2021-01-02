import os
import sys
import time
import numpy as np
import cpu_nms


def get_proposals(proposal_path):
	proposals = []
	with open(proposal_path) as f:
		line = f.readline().strip()
		while line:
			score, y1, x1, y2, x2 = line.split(',')
			score, y1, x1, y2, x2 = float(score), float(y1), float(x1), float(y2), float(x2)
			proposals.append([score, y1,x1,y2,x2])
			line = f.readline().strip()
	proposals = np.array(proposals, dtype = np.float32)
	return proposals


if __name__ == '__main__':
	proposal_path = sys.argv[1]
	proposals = get_proposals(proposal_path)
	#print(proposals.dtype)
	s = time.time()
	keep = cpu_nms.py_cpu_nms(proposals, thresh = 0.5)
	e = time.time()
	print('Time Dur: ', e-s, ' keep: ', keep, len(keep))
	