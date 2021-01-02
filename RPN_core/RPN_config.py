
Config = dict(
		base_dir = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git/',
		train_data_folder = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup/dataset/train/image_xml_resize/',
		validate_data_folder = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup/dataset/val/image_xml_resize/',
		train_steps = 200,
		learning_rate_init = 0.001,
		decay_steps = 120000,
		decay_rate = 0.1,
		stair_case = True
)