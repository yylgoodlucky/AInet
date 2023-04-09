import librosa
import os, pickle, argparse, cv2
import numpy as np
import torch
import torch.utils.data as data
import python_speech_features

from tqdm import tqdm
from glob import glob
import python_speech_features
from torchvision import transforms

import sys
sys.path.append("..")
from utils import get_mfcc_seq, get_img_list
import pdb
    
    
    
class AInet_dataset(data.Dataset):
    def __init__(self, dataset_dir, split, window_size, step):
        self.dataset_dir = dataset_dir
        self.split = split
        self.window_size = window_size
        self.step = step
        self.img_list = get_img_list(self.dataset_dir, self.split)   # (43370, )
        self.mfcc_seq = get_mfcc_seq(self.dataset_dir, self.split) 

        self.input_mfcc_all = np.concatenate(self.mfcc_seq, axis=0)  # (43225, 28, 12)
        self.img_full = np.array(self.img_list)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        input_mfcc = self.input_mfcc_all[index*self.step: index*self.step + self.window_size, :, :]
        img_window = self.img_full[index*self.step: index*self.step + self.window_size]

        num = np.random.randint(0, self.window_size)
        single_img = cv2.imread(img_window[num])
        
        img_list = []
        for i in range(self.window_size):
            img_s = self.transform((cv2.imread(img_window[i])))
            
            img_s = np.array(img_s)
            img_list.append(img_s)
            
            if i == self.window_size - 1:
                img_seq = np.concatenate((img_list[0][np.newaxis, :], img_list[1][np.newaxis, :], 
                                        img_list[2][np.newaxis, :], img_list[3][np.newaxis, :], 
                                        img_list[4][np.newaxis, :]), axis=0)
                
                # path = "/data/users/yongyuanli/workspace/myspace/AInet_0403/example/sample"
                # if not os.path.exists(path): os.mkdir(path)
                # t,b,h,w = img_seq.shape
                # img_save = img_seq.reshape(h*t,w,b).transpose(1,0,2) * 255
                # pdb.set_trace()
                # cv2.imwrite(os.path.join(path, 'gt.jpg'), img_save)
    
        single_img = self.transform(single_img)
        
        return single_img, input_mfcc, img_seq
        
    
    def __len__(self):
        return int((self.input_mfcc_all.shape[0] - self.window_size) / self.step)


        
