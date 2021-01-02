
class Config(object):
	def __init__(self):
		self.base_dir = '/root/Faster_RCNN_tensorflow_iterative_train_tidyup_git/'
		self.class_name_file = os.path.join(self.base_dir, 'dataset/class_names.names')
		self.backbone_model_path = os.path.join(self.base_dir, 'checkpoints/backbone_checkpoints/step4/backbone.ckpt-best')
		self.RPN_model_path = os.path.join(self.base_dir, 'checkpoints/RPN_checkpoints/step3/model_RPN.ckpt-best')
		self.FastRCNN_model_path = os.path.join(self.base_dir, 'checkpoints/Fast_RCNN_checkpoints/step4/fast_rcnn.ckpt-best')
		self.RPN_k = 12
		self.n_classes = 80
		self.box_num = 300
		self.image_h = 640
		self.image_w = 640
