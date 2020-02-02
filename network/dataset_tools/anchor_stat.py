import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_stat(root_dir):
    img_list = glob.glob(os.path.join(root_dir, '*.png'))
    stat_vec = np.zeros(len(img_list))
    height_vec = np.zeros(len(img_list))
    for i, img_path in tqdm(enumerate(img_list)):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        height_vec[i] = h
        stat_vec[i] = h/float(w)
    _ = plt.hist(height_vec, bins='auto')
    plt.show()


if __name__ == '__main__':
    root_dir = '/Users/zheyu/dev/rvss_dl/rvss_data/train/snake'
    get_stat(root_dir)

