import argparse, os, cv2, librosa
import face_detection
import face_alignment
import numpy as np
import python_speech_features
import torch
import pdb

from model.AInet import AInet_Generator
from torchvision import transforms
from utils import trans_to_cuda

if not os.path.isfile('./face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
                            before running this script!')

fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image", type=str, help="root folder of your inferece image", required=True)
    parser.add_argument("--audio", type=str, help="root folder of your inferece audio", required=True)
    parser.add_argument("--checkpoint_path", type=str, help="root folder of your pretrain model", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--temp", type=str, help="root folder of your predict image", required=True)

    return parser.parse_args()

config = parse_args()

# def crop_image(config):
#     img = cv2.imread(config.image)
#     preds = fa.get_detections_for_batch(np.asarray(img))

#     for j, f in enumerate(preds):
#         i += 1
#         if f is None:
#             continue

#         x1, y1, x2, y2 = f
#         padlen = (max(x2-x1, y2-y1) - min(x2-x1, y2-y1)) / 2
#         x1 = int(x1 - padlen)
#         x2 = int(x2 + padlen)
        
#         image_size = cv2.resize(img[y1:y2, x1:x2], (config.image_size, config.image_size))
        
#     return image_size

def get_mfcc(config):
    speech, sr = librosa.load(config.audio, sr=16000, mono=True)
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
        
    return input_mfcc

def animate(pred, config, num):
    print("write fake image {:05d} - {:05d} ".format(num * config.window_size, num * config.window_size + 5))
    pred = torch.squeeze(pred, dim=0)
    
    pic_base = os.path.basename(config.image).split(".")[0]
    pic_path = os.path.join(config.temp, pic_base)
    if not os.path.exists(pic_path): 
        os.mkdir(pic_path)
        
    for i in range(pred.shape[0]):
        # fake_img = (((pred[i, :] * std) + mean)).numpy().astype(int)
        fake_img = (pred[i, :].numpy() * 255).astype(int)
        cv2.imwrite("{}/{}".format(pic_path, (str(num*config.window_size+i)).zfill(5) + ".png"), np.transpose(fake_img, (1,2,0)))

    
    
def load_model(config):
    model = AInet_Generator()
    print("Load checkpoint from: {}".format(config.checkpoint_path))
    checkpoint = torch.load(config.checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s)

    model = model.cuda()
    return model.eval()

def main():
    model = load_model(config)
    print("Model loaded")

    mfcc = torch.tensor(get_mfcc(config))
    img = cv2.imread(config.image)   # h, w, 3
    
    in_img = transforms.ToTensor()(img).unsqueeze(0).float().cuda()
    
    # 归一化
    # mean = torch.mean(input_img)
    # std = torch.std(input_img)
    # input_img = ((input_img - mean) / std).cuda()

    padding = torch.tensor(np.zeros([config.window_size - (mfcc.shape[0] % 2), mfcc.shape[1], mfcc.shape[2]]))
    mfcc = torch.cat((mfcc, padding), 0).unsqueeze(0)
    
    for i in range(mfcc.shape[1]//5):
        in_mfcc = mfcc[:,i * config.window_size: i * config.window_size + 5, :, :].float().cuda()
        with torch.no_grad():
            pred = model(in_img, in_mfcc).cpu()
            
        # write_img(pred, config.temp, i, mean, std)
        animate(pred, config, i)
        
    cmd = "ffmpeg -y -loglevel warning -thread_queue_size 8192 -i {} " + \
    "-thread_queue_size 8192 -i {}/%05d.png " + \
    "-vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest {}/{}.mp4"

    pic_base = os.path.basename(config.image).split(".")[0]
    pic_path = os.path.join(config.temp, pic_base)

    os.system(cmd.format(config.audio, pic_path, config.temp, pic_base))
    print("=== Animate over ===")


if __name__=="__main__":
    main()
    