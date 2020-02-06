import sys
import os
import json
import numpy as np
sys.path.insert(0, "../")
sys.path.insert(0, "../slam")
from SlamMap import SlamMap
from Triangulate import triangulate

def loadDataset(dataDir,slamFile='slam.txt',bearingFile='bearings.txt'):

    #load slam map
    slamFilePath=dataDir + '/' + slamFile
    slammap=SlamMap()
    slammap.load(slamFilePath)

    print(slammap.markers[0])

    #load bearings
    with open('../system_output/bearings.txt') as f:
       bearing_dict = json.load(f)

    ConvertBearingDict(bearing_dict)

    print(bearing_dict[0])

    return slammap, bearing_dict

def ConvertBearingDict(bearing_dict):
    for bearing in bearing_dict:
        bearing['pose']=np.array(bearing['pose']).reshape(3,1)

def ConvertBearing(bearing,R,t):
        homoT = np.zeros((3,3))
        homoT[0][0:2] = R[0][0:2]
        homoT[1][0:2] = R[1][0:2]
        homoT[0][2] = t[0]
        homoT[1][2] = t[1]
        homoT[2][2] = 1
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
        #bearing['pose']= t + bearing['pose']
        #bearing['theta']= R
        print(bearing['pose'])

def combineBearingDicts(slammap1,bearing_dict1,slammap2,bearing_dict2):
    #Converts and fuses bearing_dict2 into slammap1 frame and appends converted bearing_dict2 resutls to bearing_dict1
    armse, R, t=slammap1.compute_tf(slammap2)

    for bearing in bearing_dict2:
        ConvertBearing(bearing,R,t)

    bearing_dict1=bearing_dict1 + bearing_dict2
    print(bearing_dict1)

if __name__ == '__main__':

    dataDir2='../system_output'
    dataDir='../someone_elses_system_output'
    slammap1, bearing_dict1 = loadDataset(dataDir)

    slammap2, bearing_dict2 = loadDataset(dataDir2)

    combineBearingDicts(slammap1,bearing_dict1,slammap2,bearing_dict2)

    for animal in ("elephant", "crocodile","llama","snake"):
        bearings = [x for x in bearing_dict1 if x["animal"] == animal]
        meas = [np.concatenate([detection["pose"].flatten(),np.array([detection["bearing"]])]) for detection in bearings]
        if len(meas) == 0:
            continue
        print('Position of ', animal)
        print(triangulate(np.array(meas)))
