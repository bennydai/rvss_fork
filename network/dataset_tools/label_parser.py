import glob
import json
import os

import cv2
import pandas as pd


def parse_label_file(csv_file, img_folder):
    df = pd.read_csv(csv_file, usecols=['Label', 'External ID'])
    row_count, _ = df.shape
    for i in range(row_count):
        img_name = df['External ID'][i]
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        #
        if df['Label'][i] != 'Skip':
            obj_struct = json.loads(df['Label'][i])
            obj_struct = obj_struct['objects']
            for i in range(len(obj_struct)):
                obj = obj_struct[i]
                label = obj['title']
                bbox = obj['bbox']
                cropped_img = bb_crop(img, bbox)
                save_img(cropped_img, './bg_temp', label)
        else:
            continue


def bb_crop(img, bbox):
    y, x = bbox['top'], bbox['left']
    h, w = bbox['height'], bbox['width']
    cropped_img = img[y: y+h, x: x+w]
    return cropped_img


def save_img(cv2_img, dataset_root, label):
    dir_path = os.path.join(dataset_root, label)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_count = len(glob.glob1(dir_path, '*.png'))
    img_name = os.path.join(dir_path, f'{file_count:06}' + '.png')
    cv2.imwrite(img_name, cv2_img)


def main():
    parse_label_file('./all_background/labels.csv', './all_background')


if __name__ == '__main__':
    main()
