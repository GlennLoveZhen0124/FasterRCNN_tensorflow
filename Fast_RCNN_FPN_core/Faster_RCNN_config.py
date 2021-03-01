
Config = dict(
		base_dir = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/',
		train_data_folder = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/dataset/train/image_xml_resize/',
		validate_data_folder = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/dataset/val/image_xml_resize/',
		train_steps = 200,
		learning_rate_init = 0.0001,
		decay_steps = 120000,
		decay_rate = 0.1,
		stair_case = True,
		class_name_file = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/dataset/class_names.names',
		width = 640,
		height = 640,
		RPN_k = 3,
		n_classes = 80
)