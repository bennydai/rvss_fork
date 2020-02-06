import evaluate
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

NUM_CLASSES = 4
PIXEL_THRESHOLD = 500


class PosedImage:
    def __init__(self, json_line):
        img_dict = json.loads(json_line)
        self.pose = np.array(img_dict["pose"])
        self.img_name = img_dict["imgfname"]
        self.class_list = self.getList()

    def getList(self):
        baconFile = open('class_list.txt')
        baconContent = baconFile.readlines()
        stuff = []
        for shit in baconContent:
            stuff.append(shit.strip())
        baconFile.close()
        return stuff
        

    def write_bearings(self, neuralnet, bearings_file, folder_name=""):
        # Obtain neural net output
        img = Image.open(folder_name+self.img_name)
        heatmap = neuralnet.sliding_window(img)

        # Compute animal bearings here and save to self.animals.
        # Next, you can use all this information to triangulate the animals!
        
        bearings = {}
        # For example, finding the llamas:
        # if np.any(heatmap == 2.0):
        #     llama_coords = np.where(heatmap == 2.0)
        #     average_llama = np.mean(llama_coords, axis=1)
        #     bearings["llama"] = ...
        # Now you need to convert this to a horizontal bearing as an angle.
        # Use the camera matrix for this!

        kernel = np.ones((3,3,), np.uint8)

        for i in range(1, NUM_CLASSES):
            if np.any(heatmap == i):
                mask = np.zeros(heatmap.shape)
                mask[heatmap == i] = 255

                mask = cv2.erode(mask,kernel,iterations = 1)

                moments = cv2.moments(mask)

                # Remove blobs that do not pass this arbitary threshold
                area = moments['m00']

                print(self.class_list[i], area)

                if area > 1:
                    # calculate x,y coordinate of center
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    print(cX, cY)
                    plt.figure(1)
                    plt.imshow(mask)
                    plt.figure(2)
                    plt.imshow(img)
                    plt.show()
                else:
                    break



        # There are ways to get much better bearings.
        # Try and think of better solutions than just averaging.

        # for animal in bearings:
        #     bearing_dict = {"pose":self.pose.tolist(),
        #                     "animal":animal,
        #                     "bearing":bearings[animal]}
        #     bearing_line = json.dumps(bearing_dict)


if __name__ == "__main__":
    # Set up the network
    exp = evaluate.Evaluate()

    # Read in the images
    images_fname = "../system_output/images.txt"
    with open(images_fname, 'r') as images_file:
        posed_images = [PosedImage(line) for line in images_file]

    # Compute bearings and write to file
    bearings_fname = "../system_output/bearings.txt"
    with open(bearings_fname, 'w') as bearings_file:
        for posed_image in posed_images:
            posed_image.write_bearings(exp, bearings_file, "../system_output/")

    