import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Import integration components
sys.path.insert(0, "{}/integration".format(os.getcwd()))
import integration.penguinPiC
import integration.DatasetHandler as dh
import control.keyboardControl as Keyboard

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
import slam.Slam as Slam
import slam.Robot as Robot
import slam.aruco_detector as aruco
import slam.Measurements as Measurements

class Operate:
    def __init__(self, datadir, ppi, writeData=False):
        # Initialise data parameters
        self.ppi = ppi
        self.ppi.set_velocity(0,0)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)

        # Set up subsystems
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(datadir)

        # Control subsystem
        self.keyboard = Keyboard.Keyboard(self.ppi)
        # SLAM subsystem
        self.pibot = Robot.Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.pibot, marker_length = 0.07)
        self.slam = Slam.Slam(self.pibot)
        
        # Optionally record input data to a dataset
        if writeData:
            self.data = dh.DatasetWriter('test')
        else:
            self.data = None
        
        self.output = dh.OutputWriter('system_output')

    def __del__(self):
        self.ppi.set_velocity(0,0)

    def getCalibParams(self,datadir):
        # Imports calibration parameters
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')

        return camera_matrix, dist_coeffs, scale, baseline

    def control(self):
        lv, rv = self.keyboard.latest_drive_signal()
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        drive_meas = Measurements.DriveMeasurement(lv, rv, dt=0.3)
        self.slam.predict(drive_meas)

    def vision(self):
        self.img = self.ppi.get_image()  
        if not self.data is None:
            self.data.write_image(self.img)
        lms,aruco_image = self.aruco_det.detect_marker_positions(self.img)       
        self.slam.add_landmarks(lms)
        self.slam.update(lms)
    
    def display(self, fig, ax):
        # Output system to screen
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:,:,-1::-1])
        
        plt.pause(0.01)
    
    def record_data(self):
        # Save data for network processing
        if self.keyboard.get_net_signal():
            self.output.write_image(self.img, self.slam)
        if self.keyboard.get_slam_signal():
            self.output.write_map(self.slam)

    def process(self):
        # Visualisation tools
        fig, ax = plt.subplots(1,2)
        img_artist = ax[1].imshow(self.img)

        # MAIN LOOP
        while True:
            # Run SLAM
            self.control()
            self.vision()

            # Save Image and/or SLAM map
            self.record_data()

            # Output visualisation
            self.display(fig, ax)


        
       
if __name__ == "__main__":   
    currentDir = os.getcwd()
    datadir = "{}/testData/testCalibration/".format(currentDir)
    
    # Use either a real or simulated penguinpi
    ppi = integration.penguinPiC.PenguinPi(ip = '192.168.50.1')
    # ppi = dh.DatasetPlayer("test")

    # Set up the integrated system
    operate = Operate(datadir, ppi, writeData=False)

    # Enter the main loop
    operate.process()



