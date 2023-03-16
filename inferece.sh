#!/bin/bash
set -e 

image=$1
audio=$2
checkpoint=$3
tem_root=/data/users/yongyuanli/workspace/Mycode/ALnet/temp

echo "python inferece_ALnet.py"
CUDA_VISIBLE_DEVICES=3 python inferece.py \ 
  --image ${image} \ 
  --audio ${audio} \ 
  --checkpoint ${checkpoint}

echo "ffmpeg -y -loglevel warning \
  -thread_queue_size 8192 -i ${audio} \
  -thread_queue_size 8192 -i ${tem_root}/landmarks/%05d.png \
  -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest ${tem_root}/output.mp4"
ffmpeg -y -loglevel warning \
  -thread_queue_size 8192 -i ${audio} \
  -thread_queue_size 8192 -i ${tem_root}/landmarks/%05d.png \
  -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest ${tem_root}/output.mp4

    