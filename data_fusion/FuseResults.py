import sys
import os
import json
import numpy as np
sys.path.insert(0, "../")
sys.path.insert(0, "../slam")
from SlamMap import SlamMap
from Triangulate import triangulate
import matplotlib.pyplot as plt
import csv

def loadDataset(dataDir,slamFile='slam.txt',bearingFile='bearings.txt'):

    #load slam map
    slamFilePath=dataDir + '/' + slamFile
    slammap=SlamMap()
    slammap.load(slamFilePath)

    #load bearings
    Filepath=dataDir+'/'+bearingFile
    with open(Filepath) as f:
       bearing_dict = json.load(f)

    print(bearing_dict)

    ConvertBearingDict(bearing_dict)

    return slammap, bearing_dict

def ConvertBearingDict(bearing_dict):
    for bearing in bearing_dict:
        bearing['pose']=np.array(bearing['pose']).reshape(3,1)

def ConvertBearing(bearing,R,t):
        #Calculate homogenous transformation matrix
        homoT = np.zeros((3,3))
        homoT[0][0:2] = R[0][0:2]
        homoT[1][0:2] = R[1][0:2]
        homoT[0][2] = t[0]
        homoT[1][2] = t[1]
        homoT[2][2] = 1
        #assemble pose matrix
        Pose=bearing['pose']
        theta=Pose[2]
        PoseRepMat=np.mat([
            [np.cos(theta), -np.sin(theta), Pose[0]],
            [np.sin(theta),  np.cos(theta), Pose[1]],
            [0,0,1]
        ])

        NewPose=homoT @ PoseRepMat
        x=NewPose[0,2]
        y=NewPose[1,2]
        thetanew=np.arccos(NewPose[0,0])
        bearing['pose']=np.array([x,y,thetanew]).reshape(3,1)
        #print(bearing['pose'])

def combineBearingDicts(slammap1,bearing_dict1,slammap2,bearing_dict2):
    #Converts and fuses bearing_dict2 into slammap1 frame and appends converted bearing_dict2 resutls to bearing_dict1
    armse, R, t=slammap1.compute_tf(slammap2)

    for bearing in bearing_dict2:
        ConvertBearing(bearing,R,t)

    return bearing_dict1 + bearing_dict2

if __name__ == '__main__':
    #Assumes results from each bot in a different directory
    #Combine all dictionaries into bearing_dict1

    dataDir2='../system_output'
    dataDir3='../SYTM_OUT_1'
    dataDir1='../system_output_16'

    slammap1, bearing_dict1 = loadDataset(dataDir1)
    #if len(dataDir2)>0:
    slammap2, bearing_dict2 = loadDataset(dataDir2)
    bearing_dict1=combineBearingDicts(slammap1,bearing_dict1,slammap2,bearing_dict2)
    #if len(dataDir3)>0:
    slammap3, bearing_dict3 = loadDataset(dataDir3)
    bearing_dict1=combineBearingDicts(slammap1,bearing_dict1,slammap3,bearing_dict3)

    fig=plt.figure()
    #ax=fig.add_axes([0,0,1,1])

    print(bearing_dict1)

    output_animal = []
    answer = []

    with open('output_answer.csv', 'wb') as output_answer:
        wr = csv.writer(output_answer, quoting=csv.QUOTE_ALL)

    for animal in ("elephant","crocodile","llama","snake"):
        bearings = [x for x in bearing_dict1 if x["animal"] == animal]
        meas = [np.concatenate([detection["pose"].flatten(),np.array([detection["bearing"]])]) for detection in bearings]
        if len(meas) == 0:
            continue
        print('Position of ', animal)
        output_animal.append(animal)
        ans=triangulate(np.array(meas))
        print(ans)
        answer.append(ans)
        plt.scatter(ans[0],ans[1],label=animal)
        plt.legend()
        plt.grid(True)
        # stuff = [animal, ans]
        # wr.writerow(stuff)


plt.show()
