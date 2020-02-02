import numpy as np
import cv2
import Measurements

class aruco_detector:
    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params)

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids:
                continue
            else:
                seen_ids.append(idi)

            lm_tvecs = tvecs[ids==idi].T
            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

            lm_measurement = Measurements.MarkerMeasurement(lm_bff2d, idi)
            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked


if __name__ == '__main__':
    import Robot

    img = cv2.imread("/home/pieter/Documents/repos/code_rvss2020_workshop/vslam/test_img.png")
    K = np.array([[256.0501596,   0.,        148.28664955],
                    [  0.,         256.0501596,  120.86729004],
                    [  0.,           0.,           1.        ]])
    dist = np.array([[-1.03571129e+00],
                    [-2.28003976e+01],
                    [-3.63713197e-03],
                    [-1.08658434e-02],
                    [ 5.00474231e+01],
                    [-1.36353362e+00],
                    [-2.07541469e+01],
                    [ 4.68844178e+01],
                    [ 0.00000000e+00],
                    [ 0.00000000e+00],
                    [ 0.00000000e+00],
                    [ 0.00000000e+00],
                    [ 0.00000000e+00],
                    [ 0.00000000e+00]])
    
    robot = Robot.Robot(0.017,0.01,K,dist)
    aruco_det = aruco_detector(robot, 0.07)
    lms,marked_img = aruco_det.detect_marker_positions(img)

    print(lms)
    cv2.imshow("markers", marked_img)
    cv2.waitKey(0)
