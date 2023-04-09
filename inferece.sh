#!/bin/bash

image="/data/users/yongyuanli/workspace/myspace/AInet_0403/example/obama.png"
audio="/data/users/yongyuanli/workspace/myspace/AInet_0403/example/moxiaomeng_girl.wav"
checkpoint="/data/users/yongyuanli/workspace/myspace/AInet_0403/checkpoint/ALnet_transform/checkpoint_epoch019_step05000.pth"
temp_root="/data/users/yongyuanli/workspace/myspace/AInet_0403/temp"

basename=$(basename $image)
ref_img_basename=${basename%.*}
echo $ref_img_basename

mkdir -p ${temp_root}/${ref_img_basename}

echo "python inferece_ALnet.py"
CUDA_VISIBLE_DEVICES=1 python inferece_img.py \ 
    --image $image \ 
    --audio $audio \ 
    --checkpoint_path $checkpoint \ 
    --temp ${temp_root}/${ref_img_basename}

echo "ffmpeg -y -loglevel warning \
  -thread_queue_size 8192 -i ${audio} \
  -thread_queue_size 8192 -i ${temp_root}/${ref_img_basename}/%05d.png \
  -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest ${temp_root}/${ref_img_basename}.mp4"
ffmpeg -y -loglevel warning \
  -thread_queue_size 8192 -i ${audio} \
  -thread_queue_size 8192 -i ${temp_root}/${ref_img_basename}/%05d.png \
  -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest ${temp_root}/${ref_img_basename}.mp4





python inferece_img.py \
--image /data/users/yongyuanli/workspace/myspace/AInet_0403/example/gt_label/00004.png \
--audio /data/users/yongyuanli/workspace/myspace/AInet_0403/example/gt_label/audio_cut.wav \
--checkpoint_path /data/users/yongyuanli/workspace/myspace/AInet_0403/checkpoint/ALnet_transform/checkpoint_epoch00_step05000.pth \
--temp /data/users/yongyuanli/workspace/myspace/AInet_0403/temp
    