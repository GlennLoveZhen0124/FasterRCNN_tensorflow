import os

Config = dict(
		base_dir = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git_extensible/',
		n_classes = 80,
		RPN_k = 3,
		width = 640,
		height = 640
)

Config['train_data_folder'] = os.path.join(Config['base_dir'], 'dataset/train/image_xml_resize/')
Config['validate_data_folder'] = os.path.join(Config['base_dir'], 'dataset/val/image_xml_resize/')
Config['train_proposal_folder_step2'] = os.path.join(Config['base_dir'], 'dataset/train/proposals_step2/')
Config['train_proposal_folder_step4'] = os.path.join(Config['base_dir'], 'dataset/train/proposals_step4/')
Config['validate_proposal_folder_step2'] = os.path.join(Config['base_dir'], 'dataset/val/proposals_step2/')
Config['validate_proposal_folder_step4'] = os.path.join(Config['base_dir'], 'dataset/val/proposals_step4/')