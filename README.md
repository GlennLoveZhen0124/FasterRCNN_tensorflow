## Faster RCNN project

#### 	Description:
    	This project use 4-steps way to train FasterRCNN model, as described in the original paper.(Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks)

		Paper address: [https://arxiv.org/abs/1506.01497](http://)

#### 	Train:
    	(1) resize your train and val images to (640,640,3), and generate corresponding xml files (VOC style)
        (2) put your train data into ./dataset/train/image_xml_resize/ folder, your validate data into ./dataset/val/image_xml_resize/ folder. (including jpg and xml files)
        (3) change config files content as you need. There are 3 config files, and they are ./RPN_FPN_core/RPN_FPN_config.py, ./Fast_RCNN_FPN_core/Faster_RCNN_config.py, ./RPN_FPN_inference/proposal_config.py
		(4) run: 
        			sh train.sh
		
#### 	TODO:
      Inference code will be uploaded soon.
    		This implementation use vgg16/resnet50 as its backbone, and FPN as detection neck.
            More backbones and detection necks such as resnest and PAN will be included in the future.
            