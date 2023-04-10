import librosa
import os, pickle, argparse, cv2, random
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
from utils import get_image_list
import pdb

class lrs2_datasets(data.Dataset):
    def __init__(self, dataset_dir, split, window_size, img_size=256):
        self.all_videos = get_image_list(dataset_dir, split)
        self.window_size = window_size
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def get_window(self, start_frame):
        start_id = int(os.path.basename(start_frame).split('.')[0])
        vidname = os.path.dirname(start_frame)
        
        window_frames = []
        for frame_id in range(start_id, start_id + self.window_size):
            frame = os.path.join(vidname, '{}.jpg'.format(frame_id))
            if not os.path.isfile(frame):
                return None
            window_frames.append(frame)
            
        return window_frames
    
    def get_input_imgs(self, window_frames):
        img_list = []
        for i in range(self.window_size):
            img_s = cv2.resize((cv2.imread(window_frames[i])), (self.img_size, self.img_size))
            img_s = np.array(self.transform(img_s))
            img_list.append(img_s)
            
            if i == self.window_size - 1:
                img_seq = np.concatenate((img_list[0][np.newaxis, :], img_list[1][np.newaxis, :], 
                                        img_list[2][np.newaxis, :], img_list[3][np.newaxis, :], 
                                        img_list[4][np.newaxis, :]), axis=0)

        return img_seq
    
    def get_single_img(self, img):
        img_s = cv2.resize((cv2.imread(img)), (self.img_size, self.img_size))
        img_s = np.array(self.transform(img_s))
        return img_s
    
    def get_input_mfcc(self, mel, img_name):
        speech, sr = librosa.load(mel, sr=16000, mono=True)
        if speech.shape[0] > 16000:
            speech = np.insert(speech, 0, np.zeros(1920))
            speech = np.append(speech, np.zeros(1920))
            mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)

            ind = 3

            input_mfcc = []
            while ind <= int(mfcc.shape[0] / 4) - 4:
                t_mfcc = mfcc[(ind - 3) * 4: (ind + 4) * 4, 1:]
                input_mfcc.append(t_mfcc)
                ind += 1

            input_mfcc = np.stack(input_mfcc, axis=0)
            idx = int(os.path.basename(img_name).split('.')[0])
            input_mfcc = input_mfcc[idx:idx+5, :, :]

            return input_mfcc
        
    def __getitem__(self, idx): 
        while 1:
            # idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(sorted(glob(os.path.join(vidname, '*.jpg'))))
            mel_path = os.path.join(vidname, 'audio.wav')
            
            img_name = random.choice(img_names[:-5])
            if img_name is None:
                continue

            window_frames = self.get_window(img_name)
            if window_frames is None or len(window_frames) < 5:
                continue

            img_seq = self.get_input_imgs(window_frames)
            
            single_img = self.get_single_img(img_name)
            
            input_mfcc = self.get_input_mfcc(mel_path, img_name)
            if input_mfcc is None or input_mfcc.shape[0] != self.window_size:
                continue
            
            return single_img, input_mfcc, img_seq
            
            
        
    def __len__(self):
        return len(self.all_videos)