import numpy as np

class MarkerMeasurement:
    # Measurements are of landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag, covariance = (0.1*np.eye(2))):
        self.position = position
        self.tag = tag
        #self.covariance=covariance
        self.covariance = self.covariance_matrix_calculation(position)

    def covariance_matrix_calculation(self, lm_bff2d, sigma_u=0.05, sigma_rho_param = 0.2):
        """ New method to initialise the covariance. Turning the x,y representation from aruco into a
            (polar) estimate of d,phi, assumed independent. Estimate sigma_rho by depth of marker
            Put this into aruco_detector.py and call in lm_measurements for the parameter covariance
        """
        # new covariance initialisation
        dist_est = np.sqrt(lm_bff2d[0]**2 + lm_bff2d[1]**2)
        u = np.divide(lm_bff2d,dist_est)
        u_mat = u @ u.T

        # rho = 1 /(dist_est)       # we never actually need rho
        cov_u = (sigma_u**2) * (np.eye(2) - u_mat)
        cov_rho = sigma_rho_param*(dist_est**2) * u_mat
        cov_full = cov_u + cov_rho
        return cov_full

class DriveMeasurement:
    # Measurement of the robot wheel velocities
    def __init__(self, left_speed, right_speed, dt, left_cov = 5, right_cov = 5):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov
