import numpy as np
import os, pickle
import face_detection
import face_alignment
import traceback
import librosa
import cv2, argparse, os, subprocess
import python_speech_features

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from glob import glob
from skimage import io
import pdb

if not os.path.isfile('./face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
                            before running this script!')
    
fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda')

audio_template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
video_template = 'ffmpeg -y -i {} -c:v libx264 -crf 18 -c:a aac -r 25 {}'



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Root folder of your dataset", required=True)
    parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
    parser.add_argument("--batch_size", help='Single GPU Face detection batch size', default=25, type=int)
    parser.add_argument("--image_size", help='Single GPU Face detection image_size', default=256, type=int)
    parser.add_argument("--window_size", help='audio mfcc for window size', default=5, type=int)

    return parser.parse_args()


def process_video_file(args):
    filelist = sorted(glob(os.path.join(args.data_root, '*.mp4')))

    for vfile in tqdm(filelist):
        vidname = os.path.basename(vfile).split('.')[0]  
        videopath = os.path.join(args.preprocessed_root, vidname, '{}.mp4'.format(vidname))

        os.makedirs(os.path.split(videopath)[0], exist_ok=True)
        command = video_template.format(vfile, videopath)
        subprocess.call(command, shell=True)
    
def video_face_detect(vfile):
    video_stream = cv2.VideoCapture(vfile)
    print(vfile)
    
    if not video_stream.open(vfile):
        print("can not open the video")
        exit(1)

    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
        
    fulldir = os.path.split(vfile)[0]
    os.makedirs(fulldir, exist_ok=True)
    
    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    
    i = -1
    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            padlen = (max(x2-x1, y2-y1) - min(x2-x1, y2-y1)) / 2
            x1 = int(x1 - padlen)
            x2 = int(x2 + padlen)
            
            image_size = cv2.resize(fb[j][y1:y2, x1:x2], (args.image_size, args.image_size))
            cv2.imwrite(os.path.join(fulldir, '{}.png'.format(str(i).zfill(5))), image_size)
            
def process_audio_file(vfile):
    fulldir = os.path.split(vfile)[0]

    wavpath = os.path.join(fulldir, 'audio.wav')

    command = audio_template.format(vfile, wavpath)
    subprocess.call(command, shell=True)

def landmark_detection(vfile):
    fulldir = os.path.split(vfile)[0]
    print(fulldir)
    image_list = sorted(glob(os.path.join(fulldir, '*.png')))
    
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda')

    landmark_list = []
    for i in tqdm(range(len(image_list))):
        image_name = image_list[i]
        image = io.imread(image_name)
        try:
            preds = fa_3d.get_landmarks(image)
        except Exception as e:
            print(f'Catched the following error: {e}')
            preds = None

        landmark = preds[0][:, :2]
        
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # landmark = scaler.fit_transform(landmark)
        landmark_list.append(landmark)
    
    with open(os.path.join(fulldir, 'landmark.pkl'), 'wb') as f:
        pickle.dump(landmark_list, f)
        
        
def get_mfcc_feature(preprocessed_root, window_size):
    file_list = sorted(glob(os.path.join(preprocessed_root, '*', '*', 'audio.wav')))
    
    for input_audio in iter(file_list):
        print("extract ...")
        fulldir = os.path.dirname(input_audio)
        speech, sr = librosa.load(input_audio, sr=16000, mono=True)
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
            
            # cut time 
            # end = int(input_mfcc.shape[0] / window_size) * window_size
            # input_mfcc = input_mfcc[:end, :, :]

        with open(os.path.join(fulldir, 'audio_feature.pkl'), 'wb') as f:
            pickle.dump(input_mfcc, f)

def main(args):
    
    # 1, change video fps...
    print("1, change video fps  ")
    try:
        process_video_file(args)
    except KeyboardInterrupt:
        exit(0)
        
    print('Started processing for {} with GPUs'.format(args.preprocessed_root))
    filelist = sorted(glob(os.path.join(args.preprocessed_root, '*', '*', '*.mp4')))
    
    ## 2, Dumping audios...
    print('2, Dumping audios...')
    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue

    # 3, Video face detection...
    print('3, Video_face_detect')
    for vfile in tqdm(filelist):
        try:
            video_face_detect(vfile)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue

    ## 4, extract landmark...
    print('4, extract landmark...')
    for vfile in tqdm(filelist):
        try:
            landmark_detection(vfile)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue
    
    ## 5, extract audio feature...
    print('5, extract audio feature...')
    get_mfcc_feature(args.preprocessed_root, args.window_size)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)