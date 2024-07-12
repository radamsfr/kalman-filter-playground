import numpy as np
from casadi import SX, MX, jacobian, sqrt, sin, cos, vertcat

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
xs = MX.sym('x')
ys = MX.sym('y')
thetas = MX.sym('theta')
betas = MX.sym('beta')
rs = MX.sym('r')

# STATE MEAN VECTOR
x = vertcat(xs, ys, thetas)
# print(x)

P = np.array([[500., 0.],
              [0., 49.]])

dt = 0.1

fxu = vertcat(xs-rs*sin(thetas) + rs*sin(thetas+betas),
              ys+rs*cos(thetas)- rs*cos(thetas+betas),
              thetas+betas)
# print(fxu)

Q = np.array([[0.588, 1.175],
              [1.175, 2.35 ]])

u=np.array([[0]])
B = np.array([0.])
H = np.array([0])
R = np.array([[5.]])



"""
# MEASUREMENT
z = np.array([[20]])  

kf = KalmanFilter(state_vars=3, output_vars=1, x=x, P=P, F=F, Q=Q, u=u, B=B, z=z, R=R, H=H)


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


"""