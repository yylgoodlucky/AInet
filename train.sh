rm -rf /data/users/yongyuanli/workspace/myspace/AInet_0403/checkpoint/ALnet_transform/*

CUDA_VISIBLE_DEVICES=1 python train.py \
--preprocessed_dir=/data/users/yongyuanli/workspace/myspace/Obama-Lip-Sync-master/preprocessedData \
--checkpoint_dir=/data/users/yongyuanli/workspace/myspace/AInet_0403/checkpoint/ALnet_transform