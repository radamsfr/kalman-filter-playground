import numpy as np

from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])

class kalman_filter():
    def __init__(self, dim_x, dim_z):
        # initialize
        print("")

        
    def update(prior, measurement):
        x, P = prior        # mean and variance of prior
        z, R = measurement  # mean and variance of measurement
        
        y = z - x        # residual
        K = P / (P + R)  # Kalman gain

        x = x + K*y      # posterior    
        P = (1 - K) * P  # posterior variance
        return gaussian(x, P)

    def predict(posterior, movement):
        x, P = posterior # mean and variance of posterior
        dx, Q = movement # mean and variance of movement
        x = x + dx
        P = P + Q
        return gaussian(x, P)