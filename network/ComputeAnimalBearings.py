import evaluate
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

NUM_CLASSES = 4

class PosedImage:
    def __init__(self, json_line):
        img_dict = json.loads(json_line)
        self.pose = np.array(img_dict["pose"])
        self.img_name = img_dict["imgfname"]
        self.class_list = self.getList()
        self.bearings_list = []

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
        image = cv2.imread(folder_name+self.img_name)

        # Loading the instrinsic and calculating scale
        instrinsic = np.loadtxt('../testData/testCalibration/intrinsic.txt', delimiter=',')
        scale = image.shape[1] / heatmap.shape[1]
        print(scale)


        # Load the instrinsic parameters
        fx = instrinsic[0][0]
        cx = instrinsic[0][2]
        fy = instrinsic[1][1]
        cy = instrinsic[1][2]

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

        print('Processing', self.img_name)

        area_h = 0
        moments_h = None
        print(image.shape)

        for i in range(1, NUM_CLASSES):
            if np.any(heatmap == i):
                mask = np.zeros(heatmap.shape)
                mask[heatmap == i] = 255

                c_mask = mask.shape[0] / 2

                moments = cv2.moments(mask)

                # Remove blobs that do not pass this arbitary threshold
                area = moments['m00']

                if area > area_h:
                    print(self.class_list[i], area)
                    area_h = area
                    chosen_class = self.class_list[i]
                    moments_h = moments

        if area_h == 0:
            return


        print('Chose ', chosen_class, ' of area:', area_h)

        # calculate x,y coordinate of center
        u = int(moments_h["m10"] / moments_h["m00"])
        v = int(moments_h["m01"] / moments_h["m00"])

        # If the image is on the left - should be negative
        if u >= c_mask:
            polarity = 1
        else:
            polarity = -1

        # Rescale u and v
        u = u * scale
        v = v * scale

        input = [u, v, 1]
        input = np.array(input).reshape((3,1))

        # Calculating the corresponding vector
        transformed = np.matmul(np.linalg.inv(instrinsic), input)
        bearing = np.arctan(transformed[0])
        print('Coordinates is ', u, v)
        print('Angle of object is ', np.rad2deg(-bearing))

        # plt.figure(1)
        # plt.imshow(mask)
        # plt.figure(2)
        # plt.imshow(img)
        # plt.show()

        # There are ways to get much better bearings.
        # Try and think of better solutions than just averaging.

        # for animal in bearings:
        #     bearing_dict = {"pose":self.pose.tolist(),
        #                     "animal":animal,
        #                     "bearing":bearings[animal]}
        #     bearing_line = json.dumps(bearing_dict)

        bearings_list = [{"pose": self.pose.tolist(),
                            "animal": chosen_class,
                            "bearing": float(bearing)}]

        bearings_file.write(json.dumps(bearings_list) + ',')

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