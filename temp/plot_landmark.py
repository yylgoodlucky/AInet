
"""
0309
plot circle in image by landmark coord
"""


import cv2, os, pickle, glob
from tqdm import tqdm
import pdb

def get_file_list(lmark):
    img_list = sorted(glob.glob(os.path.join(lmark, "*.jpg")))
    return img_list

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def plot_landmark(data_dir, lmark):
    create_dir(os.path.join(data_dir, 'landmark'))

    with open(os.path.join(lmark, 'landmark.pkl'), 'rb') as f:
        landmark_list = pickle.load(f)

    image_list = get_file_list(lmark)

    i = 0
    for image_name in tqdm(image_list):
        image = cv2.imread(image_name)
        landmark = landmark_list[i]
        i +=1

        # pdb.set_trace()
        for point in landmark:
            image = cv2.circle(image, (int(point[0]), int(point[1])), radius=0, color=(255, 0, 0), thickness=-1)

        cv2.imwrite(os.path.join(data_dir, 'landmark', os.path.basename(image_name)), image)

def main():
    data_dir = "/data/users/yongyuanli/workspace/Mycode/ALnet/temp"
    lmark = "/data/users/yongyuanli/workspace/Mycode/Obama-Lip-Sync-master/preprocessedData/test_datasets/11"
    plot_landmark(data_dir, lmark)
    
    
if __name__=="__main__":
    main()