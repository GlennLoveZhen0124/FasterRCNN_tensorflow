python3 ./RPN_FPN_core/RPN_FPN_train.py -step=1 -pretrain=resnet_v1_50.ckpt -gpu=0
python3 ./RPN_FPN_inference/generate_FPN_proposals.py -step='inference2' -gpu=0
python3 ./Fast_RCNN_FPN_core/Faster_RCNN_FPN_train.py -step=2 -pretrain=resnet_v1_50.ckpt -gpu=0
python3 ./RPN_FPN_core/RPN_FPN_train.py -step=3 -gpu=0
python3 ./RPN_FPN_inference/generate_FPN_proposals.py -step='inference4' -gpu=0
python3 ./Fast_RCNN_FPN_core/Faster_RCNN_FPN_train.py -step=4 -gpu=0