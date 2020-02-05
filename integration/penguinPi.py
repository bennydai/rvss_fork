import numpy as np
import requests
import cv2      

ip = '192.168.50.1'

def set_velocity(vel0, vel1, time=0):
    if time == 0:
        r = requests.get("http://"+ip+":8080/robot/set/velocity?value="+str(vel0)+","+str(vel1))
    else:
        assert (time > 0), "Time must be positive."
        assert (time < 30), "Time must be less than network timeout (20s)."
        r = requests.get("http://"+ip+":8080/robot/set/velocity?value="+str(vel0)+","+str(vel1)
                        +"&time="+str(time))
       
def get_image():
    r = requests.get("http://"+ip+":8080/camera/get")
    img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)

    return img

def inverse_kinematics(baseline, scale, lin_velocity, ang_velocity):
    orientation = np.array([[1, -baseline/2],[1, baseline/2]])
    vel = np.array([[lin_velocity],[ang_velocity]])
    return scale * orientation.dot(vel)
        
