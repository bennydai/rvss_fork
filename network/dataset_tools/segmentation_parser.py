import json
import os
import shutil
import urllib.request

import pandas as pd
from tqdm import tqdm


def parse_label_file(img_folder, dest_root):
    csv_file = os.path.join(img_folder, 'masks.csv')
    df = pd.read_csv(csv_file, usecols=['Label', 'External ID'])
    row_count, _ = df.shape
    for i in tqdm(range(row_count)):
        img_name = df['External ID'][i]
        img_path = os.path.join(img_folder, img_name)
        if df['Label'][i] != 'Skip':
            obj_struct = json.loads(df['Label'][i])
            mask_obj = obj_struct['objects'][0]
            mask_url = mask_obj['instanceURI']
            obj_label = mask_obj['title']
            img_dest_folder = os.path.join(dest_root, obj_label, 'images')
            check_folder(img_dest_folder)
            shutil.copy(img_path, os.path.join(img_dest_folder, img_name))
            mask_folder = os.path.join(dest_root, obj_label, 'masks')
            check_folder(mask_folder)
            urllib.request.urlretrieve(mask_url, os.path.join(
                mask_folder, img_name))
        else:
            continue


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    parse_label_file('./mugshots', './mugshots_masked')
