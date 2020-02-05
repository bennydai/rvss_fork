import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def gnrt_data(root_dir, dest_dir, aug_ratio):
    bg_list = glob.glob(os.path.join('./bg_temp/background', '*'))
    img_list = glob.glob(os.path.join(root_dir, 'images', '*.png'))
    print(os.path.join(root_dir, 'images', '*.png'))
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        mask = cv2.imread(img_path.replace('images', 'masks'))
        img_crop, mask_crop = crop_object(img, mask)
        for i in range(aug_ratio):
            rand_bg_idx = np.random.randint(len(bg_list))
            bg = cv2.imread(bg_list[rand_bg_idx])
            img_out = add_background(img_crop, mask_crop, bg)
            file_count = len(glob.glob1(dest_dir, '*.png'))
            out_path = os.path.join(dest_dir,
                                    'night_'+f'{file_count:06}' + '.png')
            cv2.imwrite(out_path, img_out)


def crop_object(img, mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 120, 1, cv2.THRESH_BINARY)
    temp = np.argwhere(binary_mask == 1)
    x_max, y_max = np.max(temp, axis=0)
    x_min, y_min = np.min(temp, axis=0)
    img_crop = img[x_min:x_max, y_min:y_max, :]
    mask_crop = binary_mask[x_min:x_max, y_min:y_max]
    return img_crop, mask_crop


def add_background(img, mask, bg):
    # pad image with random border
    h, w, _ = img.shape
    # randomly generate number of paddings based on scale
    scale = np.random.uniform(low=0.8, high=1.0)
    h_out = int(np.ceil(h/scale))
    w_out = int(np.ceil(w/scale))
    top_pad = np.random.randint(h_out-h)
    left_pad = np.random.randint(w_out - w)
    out_img = np.zeros((h_out, w_out, 3))
    out_mask = np.zeros((h_out, w_out))
    out_img[top_pad: top_pad+h, left_pad: left_pad+w, :] = img
    out_img = out_img.astype(np.uint8)
    out_mask[top_pad: top_pad+h, left_pad: left_pad+w] = mask
    out_mask = out_mask.astype(np.uint8)
    bg = cv2.resize(bg, (w_out, h_out))
    masked_img = cv2.bitwise_and(out_img, out_img, mask=out_mask)
    masked_bg = cv2.bitwise_and(bg, bg, mask=(1-out_mask))
    out = masked_img + masked_bg
    resize_ratio = np.random.uniform(low=0.1, high=1)
    out = cv2.resize(out, (max(16, int(w_out*resize_ratio)),
                     max(16, int(h_out*resize_ratio))))
    return out


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    root_dir = './mugshots_masked/elephant'
    dest_dir = '../rvss_data/train/elephant'
    # check_folder(dest_dir)
    gnrt_data(root_dir, dest_dir, 20)

