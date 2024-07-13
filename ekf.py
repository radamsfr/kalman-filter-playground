import numpy as np
# import casadi as ca
from casadi import MX, jacobian, sqrt, sin, cos, arctan2, vcat, hcat

class KalmanFilter():
    def __init__(self, state_vars, output_vars, x, P, fxu, Q, u, B, z, R, hx):
        # initialize
        self.x = x  # STATE MEAN
        self.P = P  # STATE COVARIANCE
        self.fxu = fxu  # STATE TRANSITION FUNCTION (PROCESS FUNCTION IN MATRIX FORM)
        self.Q = Q  # PROCESS COVARIANCE (NOISE)
        
        self.u = u  # CONTROL INPUT
        self.B = B  # CONTROL FUNCTION
        
        self.z = z  # MEASUREMENT MEAN
        self.R = R  # MEASUREMENT COVARIANCE
        self.hx = hx  # MEASUREMENT TRANSITION FUNCTION
        self.y = None  # RESIDUAL (INITIALIZED LATER)
        self.K = None  # KALMAN GAIN (INITIALIZED LATER)
        

        
    def all_variables(self):
        print(f"x: {self.x}\nP: {self.P}\nfxu: {self.fxu}\nQ: {self.Q}\nu: {self.u}\nB: {self.B}\nhx: {self.hx}\nz: {self.z}\nR: {self.R}\ny: {self.y}\nK: {self.K}")

        
    def update(self, z):
        self.z = z
        
        H = jacobian(self.hx, self.x)

        self.y = self.z - self.hx(self.x)
        
        Ht = self.H.T
        S = H @ self.P @ Ht + self.R
        self.K = self.P @ Ht @ np.linalg.inv(S)
        
        self.x = self.x + self.K @ self.y

        I = np.eye(self.P.shape[0])  # Identity matrix
        self.P = (I - self.K @ self.H) @ self.P
        
        
        return (self.x, self.P)
        

    def predict(self):
        F = jacobian(self.fxu, self.x)
        V = jacobian(self.fxu, self.u)
        
        self.x = self.fxu(self.x, self.u)
        
        self.P = F @ self.P @ F.T + V @ self.Q @ V.T
        
        return (self.x, self.P)


# Define the state transition function (process model)
def fxu(x, u):
    T = u[0]  # Time step
    omega = u[1]  # Turn rate

    px, py, theta, v = x[0], x[1], x[2], x[3]
    
    if omega != 0:
        px_new = px + (v/omega) * (sin(theta + omega * T) - sin(theta))
        py_new = py + (v/omega) * (cos(theta) - cos(theta + omega * T))
    else:
        px_new = px + v * T * cos(theta)
        py_new = py + v * T * sin(theta)
    
    theta_new = theta + omega * T
    v_new = v

    return vcat([px_new, py_new, theta_new, v_new])

# Define the measurement function
def hx(x):
    px, py = x[0], x[1]
    
    r = sqrt(px**2 + py**2)
    phi = arctan2(py, px)
    
    return vcat([r, phi])


# Initial state
x0 = np.array([0.0, 0.0, 0.0, 1.0])

# Initial covariance
P0 = np.eye(4)

# Process noise covariance
Q = np.eye(4) * 0.1

# Measurement noise covariance
R = np.eye(2) * 0.1

# Control input (time step and turn rate)
u = np.array([1.0, 0.1])

# Measurement
z = np.array([1.1, 0.1])

kf = KalmanFilter(x0, P0, fxu, Q, u, None, z, R, hx)

# Run the prediction step
x_pred, P_pred = kf.predict()
print("Predicted state:", x_pred)
print("Predicted covariance:", P_pred)

# Run the update step
x_upd, P_upd = kf.update(z)
print("Updated state:", x_upd)
print("Updated covariance:", P_upd)

""" 

# TESTING
xs = MX.sym('x')
ys = MX.sym('y')
vs = MX.sym('v')
thetas = MX.sym('theta')
alphas = MX.sym('alpha')
betas = MX.sym('beta')
rs = MX.sym('r')

pxs = MX.sym('p_x')
pys = MX.sym('p_y')

# STATE MEAN VECTOR
x = vcat([xs, ys, thetas])
# print("x:\n", x)

# STATE COVARIANCE
P = np.array([[500., 0.],
              [0., 49.]])

# PROCESS MODEL FUNCTION MATRIX
fxu = vcat([xs-rs*sin(thetas) + rs*sin(thetas+betas),
              ys+rs*cos(thetas)- rs*cos(thetas+betas),
              thetas+betas])
# print("fxu:\n", fxu)

# PROCESS COVARIANCE (NOISE)
Q = np.array([[0.588, 1.175],
              [1.175, 2.35 ]])



# CONTROL INPUT
u=np.array(vcat([vs, alphas]))

# CONTROL FUNCTION
B = np.array([0.])



# MEASUREMENT TRANSITION FUNCTION
hx = vcat([sqrt((pxs-xs)**2 + (pys-ys)**2),
            arctan2(pys-ys, pxs-xs) - thetas])

# MEASUREMENT COVARIANCE
R = np.array([[0.588, 1.175],
              [1.175, 2.35 ]])



# MEASUREMENT
z = vcat([sqrt((pxs-xs)**2 + (pys-ys)**2),
            arctan2(pys-ys, pxs-xs) - thetas])

kf = KalmanFilter(state_vars=3, output_vars=1, x=x, P=P, fxu=fxu, Q=Q, u=u, B=B, z=z, R=R, hx=hx)


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