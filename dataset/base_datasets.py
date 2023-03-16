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
from utils import get_landmark_seq, get_mfcc_seq, get_img_list
    
    
    
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
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        input_mfcc = self.input_mfcc_all[index*self.step: index*self.step + self.window_size, :, :]
        img_window = self.img_full[index*self.step: index*self.step + self.window_size]

        num = np.random.randint(0, self.window_size)
        single_img = cv2.imread(img_window[num])
        
        img_seq = []
        for i in range(self.window_size):
            img_s = self.transform((cv2.imread(img_window[i])))
            img_s = np.array(img_s)
            img_seq.append(img_s)
        
        img_seq = np.array(img_seq)
        single_img = self.transform(single_img)
        
        return single_img, input_mfcc, img_seq
        
    
    def __len__(self):
        return int((self.input_mfcc_all.shape[0] - self.window_size) / self.step)


        
