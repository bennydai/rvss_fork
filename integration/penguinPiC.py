import numpy as np
import requests
import cv2      

class PenguinPi:
    def __init__(self, ip = '192.168.50.1'):
        self.ip = ip

    def set_velocity(self, lvel, rvel, time=0):
        if time == 0:
            r = requests.get("http://"+self.ip+":8080/robot/set/velocity?value="+str(lvel)+","+str(rvel))
        else:
            assert (time > 0), "Time must be positive."
            assert (time < 30), "Time must be less than network timeout (20s)."
            r = requests.get("http://"+self.ip+":8080/robot/set/velocity?value="+str(lvel)+","+str(rvel)
                            +"&time="+str(time))
        return lvel, rvel
        
    def get_image(self):
        try:
            r = requests.get("http://"+self.ip+":8080/camera/get", timeout=1.0)
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
        return img
            
