# Audio2landmark network(ALnet) && Audio2image network(AInet) by yyl 

### upgrade 20230316

### step1: preprocess data
python preprocess.py --data_root your_origin_data --preprocessed_root your_preprocessed_root

### step2: train model
python train.py --preprocessed_root your_preprocessed_root

### step3: inferece
sh inferece.sh# AInet
