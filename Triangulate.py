import evaluate
from PIL import Image
import json
import numpy as np


class PosedImage:
    def __init__(self, json_line):
        img_dict = json.loads(json_line)
        self.pose = np.array(img_dict["pose"])
        self.img_name = img_dict["imgfname"]
        self.animals = []
    
    def compute_bearing(self, neuralnet, folder_name=""):
        # Obtain neural net output
        heatmap = neuralnet.sliding_window(folder_name+self.img_name)

        # Compute animal bearings here and save to self.animals.
        # Next, you can use all this information to triangulate the animals!
    




if __name__ == "__main__":
    exp = evaluate.Evaluate()
    # img = 'dataset_tools/pibot_data_01/55.png'
    # heat_map = exp.sliding_window(img)
    # exp.visualise_heatmap(heat_map, Image.open(img))

    data_fname = "../system_output/images.txt"
    with open(data_fname, 'r') as data_file:
        posed_images = [PosedImage(line) for line in data_file]

    posed_images[0].compute_bearing(exp, "../system_output/")

    print(posed_images[0].img_name)
    print("good")
