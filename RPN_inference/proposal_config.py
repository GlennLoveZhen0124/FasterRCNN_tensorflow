import os

Config = dict(
		base_dir = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git/',
		n_classes = 80,
		RPN_k = 12,
		width = 640,
		height = 640
)

Config['train_data_folder'] = os.path.join(Config['base_dir'], 'dataset/train/image_xml_resize/')
Config['validate_data_folder'] = os.path.join(Config['base_dir'], 'dataset/val/image_xml_resize/')