import numpy as np

def eta(alpha):
    return np.mat([
        [-np.sin(alpha)],
        [np.cos(alpha)]
    ])

def rpk(theta):
    return np.mat([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def triangulate(meas):
    #Measurement is of the format list of [x,y,theta,alpha]
    n = np.shape(meas)[0]
    xpk   = meas[:,0:2]
    theta = meas[:,2]
    alpha = meas[:,3]
    A = np.zeros((n, 2))
    b = np.zeros((n, 1))
    for i in range(n):
        A[i] = np.transpose(eta(alpha[i])) @ np.transpose(rpk(theta[i]))
        b[i] = A[i] @ xpk[i]
    return np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ b

if __name__ == '__main__':
    meas = np.array([
        [0, 0, np.radians(  0), 0],
        [0, 2, np.radians(315), 0],
        [2, 0, np.radians(179), 0]
    ])
    print(triangulate(meas))
