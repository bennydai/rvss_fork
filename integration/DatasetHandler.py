import numpy as np
import cv2
import time
import csv
import os
import json

class DatasetWriter:
    def __init__(self, dataset_name):
        self.folder = dataset_name+'/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        kb_fname = self.folder + "keyboard.csv"    
        self.kb_f = open(kb_fname, 'w')
        self.kb_fc = csv.writer(self.kb_f)
        
        img_fname = self.folder + "images.csv"
        self.img_f = open(img_fname, 'w')
        self.img_fc = csv.writer(self.img_f)
        
        self.t0 = time.time()
        self.image_count = 0
    
    def __del__(self):
        self.kb_f.close()
        self.img_f.close()
    
    def write_keyboard(self, left_vel, right_vel):
        ts = time.time() - self.t0
        row = [ts, left_vel, right_vel]
        self.kb_fc.writerow(row)
        self.kb_f.flush()
    
    def write_image(self, image):
        ts = time.time() - self.t0
        img_fname = self.folder+str(self.image_count)+".png"
        row = [ts, self.image_count, img_fname]

        self.img_fc.writerow(row)
        self.img_f.flush()

        cv2.imwrite(img_fname, image)
        self.image_count += 1



class DatasetPlayer:
    def __init__(self, dataset_name):
        self.kb_f = open(dataset_name+"/"+"keyboard.csv", 'r')
        self.kb_fc = csv.reader(self.kb_f)

        self.img_f = open(dataset_name+"/"+"images.csv", 'r')
        self.img_fc = csv.reader(self.img_f)

        self.t0 = time.time()
    
    def __del__(self):
        self.f.close()
    
    def get_image(self):
        try:
            row = next(self.img_fc)
        except StopIteration:
            print("End of image data.")
            while True:
                time.sleep(1)

        t = float(row[0])
        img = cv2.imread(row[2])
        while time.time() - self.t0 < t:
            continue
        return img
    
    def set_velocity(self, left_vel, right_vel):
        try:
            row = next(self.kb_fc)
        except StopIteration:
            print("End of keyboard data.")
            while True:
                time.sleep(1)

        t = float(row[0])
        while time.time() - self.t0 < t:
            continue
        lv, rv = float(row[1]), float(row[2])
        return lv, rv


class OutputWriter:
    def __init__(self, folder_name="output/"):
        if not folder_name.endswith("/"):
            folder_name = folder_name + "/"
        self.folder = folder_name
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        self.img_f = open(folder_name+"images.txt", 'w')   
        self.map_fn = folder_name+"slam.txt"

        self.image_count = 0
        
    def __del__(self):
        self.img_f.close()
        self.map_f.close()
    
    def write_map(self, slam):
        map_dict = {"taglist":slam.taglist,
                    "map":slam.markers.tolist(),
                    "covariance":slam.P[3:,3:].tolist()}
        with open(self.map_fn, 'w') as map_f:
            json.dump(map_dict, map_f, indent=2)
            
    def write_image(self, image, slam):
        img_fname = "{}.png".format(self.image_count)
        self.image_count += 1
        img_dict = {"pose":slam.robot.state.tolist(),
                    "imgfname":img_fname}
        img_line = json.dumps(img_dict)
        self.img_f.write(img_line+'\n')
        self.img_f.flush()
        cv2.imwrite(img_fname, image)


if __name__ == '__main__':
    # For testing
    ds = DatasetWriter("test")

    for _ in range(100):
        img = np.random.randint(0,255,(240,320))
        
        ds.write_keyboard(1,2)
        ds.write_image(img)
        time.sleep(0.1)



    
