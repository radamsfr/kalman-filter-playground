import numpy as np

# from filterpy.common import Q_discrete_white_noise

from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])

class KalmanFilter():
    def __init__(self, state_vars, output_vars, x, P, F, Q, u, B, z, R, H):
        # initialize
        self.x = x  # STATE MEAN
        self.P = P  # STATE COVARIANCE
        self.F = F  # STATE TRANSITION FUNCTION (PROCESS FUNCTION IN MATRIX FORM)
        self.Q = Q  # PROCESS COVARIANCE (NOISE)
        
        self.u = u  # CONTROL INPUT
        self.B = B  # CONTROL FUNCTION
        
        self.z = z  # MEASUREMENT MEAN
        self.R = R  # MEASUREMENT COVARIANCE
        self.H = H  # MEASUREMENT TRANSITION FUNCTION
        self.y = None  # RESIDUAL (INITIALIZED LATER)
        self.K = None  # KALMAN GAIN (INITIALIZED LATER)
        

        
    def all_variables(self):
        print(f"x: {self.x}\nP: {self.P}\nF: {self.F}\nQ: {self.Q}\nu: {self.u}\nB: {self.B}\nH: {self.H}\nz: {self.z}\nR: {self.R}\ny: {self.y}\nK: {self.K}")

        
    def update(self, z):
        self.z = z

        self.y = self.z - self.H @ self.x
        
        Ht = self.H.T
        S = self.H @ self.P @ Ht + self.R
        self.K = self.P @ Ht @ np.linalg.inv(S)
        
        self.x = self.x + self.K @ self.y
        

        I = np.eye(self.P.shape[0])  # Identity matrix
        self.P = (I - self.K @ self.H) @ self.P
        
        
        return (self.x, self.P)
        

    def predict(self):
        self.x = self.F @ self.x + self.B @ self.u
        
        Ft = self.F.T
        self.P = self.F @ self.P @ Ft + self.Q


        return (self.x, self.P)




# TESTING

x = np.array([[10.0],
              [4.5]])

P = np.array([[500., 0.],
              [0., 49.]])

dt = 0.1
F = np.array([[1, dt],
              [0, 1]])

Q = np.array([[0.588, 1.175],
              [1.175, 2.35 ]])

u=np.array([[0]])
B = np.array([0.])

H = np.array([[1., 0.]])
R = np.array([[5.]])



# MEASUREMENT
z = np.array([[20]])  

kf = KalmanFilter(state_vars=2, output_vars=1, x=x, P=P, F=F, Q=Q, u=u, B=B, z=z, R=R, H=H)


# Predict step
print("PREDICTION:")
x, P = kf.predict()
print('x =', x)
print('P =', P)

# Update step with a new measurement
print("\nUPDATE:")
x, P = kf.update(z)
print('x =', x)

# Print all variables
# kf.all_variables()


# print(k.all_variables())