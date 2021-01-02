python3 ./RPN_core/RPN_train.py -step=1 -pretrain=vgg_16.ckpt -gpu=0
python3 ./RPN_inference/generate_proposals.py -step=2 -gpu=0
python3 ./Fast_RCNN_core/Faster_RCNN_train.py -step=2 -pretrain=vgg_16.ckpt -gpu=0
python3 ./RPN_core/RPN_train.py -step=3 -gpu=0
python3 ./RPN_inference/generate_proposals.py -step=4 -gpu=0
python3 ./Fast_RCNN_core/Faster_RCNN_train.py -step=4 -gpu=0