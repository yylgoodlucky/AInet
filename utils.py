import os, pickle, random, cv2
from glob import glob
import numpy as np
import librosa
import python_speech_features
import torch
import face_alignment
from matplotlib import pyplot as plt

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def plot_landmarks(landmark, image, save_path, num):
    # try:
    dpi = 100
    img = cv2.imread(image)
    fig = plt.figure(figsize=(img.shape[1]/dpi, img.shape[0]/dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    ax.imshow(np.ones(img.shape))
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    #chin
    plt.plot(landmark[0:17,0],landmark[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
    #left and right eyebrow
    plt.plot(landmark[17:22,0],landmark[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
    plt.plot(landmark[22:27,0],landmark[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
    #nose
    plt.plot(landmark[27:31,0],landmark[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
    plt.plot(landmark[31:36,0],landmark[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
    #left and right eye
    plt.plot(landmark[36:42,0],landmark[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
    plt.plot(landmark[42:48,0],landmark[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
    #outer and inner lip
    plt.plot(landmark[48:60,0],landmark[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
    plt.plot(landmark[60:68,0],landmark[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2)

    fig.canvas.draw()
    
    plt_landmarks_dir = os.path.join(save_path, "plot_landmarks")
    if not os.path.exists(plt_landmarks_dir):
        os.makedirs(plt_landmarks_dir)
    
    for n in range(5):
        fig.savefig(os.path.join(plt_landmarks_dir, "{:05d}.png".format(num*5-4+n)))    # num*window_size - overlay
        n += 1

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_epoch{:02d}_step{:05d}.pth".format(epoch, step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "step": step,
        "epoch": epoch,
    }, checkpoint_path)
    
    print("Saved checkpoint:", checkpoint_path)

def get_img_list(preprocessed_root, split):
    if split == "train":
        file_list = sorted(glob(os.path.join(preprocessed_root, 'train_datasets', '*', '*.png')))
    if split == "test":
        file_list = sorted(glob(os.path.join(preprocessed_root, 'test_datasets', '*', '*.png')))
    
    return file_list

def get_mfcc_seq(preprocessed_root, split):
    if split == "train":
        file_list = sorted(glob(os.path.join(preprocessed_root, 'train_datasets', '*', 'audio_feature.pkl')))
    if split == "test":
        file_list = sorted(glob(os.path.join(preprocessed_root, 'test_datasets', '*', 'audio_feature.pkl')))

    mfcc_list = []
    for audio_feature in iter(file_list):
        with open (audio_feature, 'rb') as f:
            audio_feature = pickle.load(f)
        mfcc_list.append(audio_feature)

    return mfcc_list
        
def main():
    preprocessed_root= "/data/users/yongyuanli/workspace/Mycode/Obama_Lip_Sync_master/preprocessedData"
    get_img_list(preprocessed_root, split="train")

if __name__ == "__main__":
    main()
    

        