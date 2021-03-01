import os
import shutil


class Utils(object):
	def __init__(self):
		pass
	
	def copy_files(self, orig_key, dst_key, folder):
		for f in os.listdir(folder):
			if orig_key in f:
				new_f = f.replace(orig_key, dst_key)
				orig = os.path.join(folder, f)
				dst = os.path.join(folder, new_f)
				shutil.copy2(orig, dst)
