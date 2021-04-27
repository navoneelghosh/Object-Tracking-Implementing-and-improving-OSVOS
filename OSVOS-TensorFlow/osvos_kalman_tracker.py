"""
Navoneel Ghosh

This is a Kalman filter implementation over the segmentation result from the OSVOS.

OSVOS is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017 
Please consider citing the original paper if you use this code.

"""

import random
import uuid
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


def getKalmanFilter(m):
    kalman = KalmanFilter(len(m) * 2, len(m))
    kalman.x = np.hstack((m, [0.0, 0.0])).astype(np.float)
    kalman.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kalman.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kalman.P *= 1000
    kalman.R = 0.00001
    kalman.Q = Q_discrete_white_noise(dim=4, dt=1, var=5)
    kalman.B = 0
    return kalman


class item:
    def __init__(self, location):
        self.id = uuid.uuid4()
        self.tracker = getKalmanFilter(location)
        self.prevLoc = [location]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update(self, measurement):
        self.tracker.predict()
        self.tracker.update(measurement)
        self.prevLoc.insert(0, self.tracker.x[:2])
        # if len(self.prevLoc) > 30:
        #     self.prevLoc.pop()
        return self.prevLoc

    def getPred(self):
        return self.tracker.get_prediction()[0][:2]
