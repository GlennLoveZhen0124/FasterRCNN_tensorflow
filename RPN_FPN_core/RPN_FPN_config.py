import os


Config = dict(
		base_dir = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/',
		train_steps = 160000,
		learning_rate_init = 0.001,
		decay_steps = 120000,
		decay_rate = 0.1,
		stair_case = True,
		image_H = 640,
		image_W = 640,
		strides = [4,8,16,32,64]
)

Config['train_data_folder'] = os.path.join(Config['base_dir'], 'dataset/train/image_xml_resize/')
Config['validate_data_folder'] = os.path.join(Config['base_dir'], 'dataset/val/image_xml_resize/')