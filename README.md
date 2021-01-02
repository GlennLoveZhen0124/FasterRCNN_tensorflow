## Faster RCNN project

#### 	Description:
    	This project use 4-steps way to train FasterRCNN model, as described in the original paper.
        (Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks)

		Paper address: https://arxiv.org/abs/1506.01497

#### 	Train:
    	First, resize your train and val images to 640x640, and generate corresponding xml files(VOC style)
        
        	Second,put your train data into ./dataset/train/image_xml_resize/ folder, your validate data into
        ./dataset/val/image_xml_resize/ folder. (including jpg and xml files)
        
        	Third, change config files content as you need. There are 3 config files, and they are 
        ./RPN_core/RPN_config.py, ./Fast_RCNN_core/Faster_RCNN_config.py, ./RPN_inference/proposal_config.py
        
		Fourth, just run: 
        		sh train.sh
		
####         Inference:
        	Just run: 
        		python3 ./inference/test.py -image=some_picture.jpg -save=some_save_name.jpg -gpu=0
#### 	TODO:
    	This is a very basic version with vgg16 as its backbone, no detection neck included.
            More backbones and detection necks such as FPN and PAN will be included in the future.
            