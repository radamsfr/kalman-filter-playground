from casadi import SX, sqrt, vcat, SX_eye
import numpy as np
from filterpy.common import Q_discrete_white_noise
import time
import matplotlib.pyplot as plt

import ekf2
import plotting

### SET VARIABLES (USING casADi SX)
x = SX.sym('x')
y = SX.sym('y')
alpha = SX.sym('alpha')
beta = SX.sym('beta')


### SET STATE MEANS AND STATE MEAN VARIABLES
x_sym = vcat([x,y])
x0 = SX([100, 100])


### SET CONTROL INPUTS AND CONTROL INPUT VARIABLES
u_sym = vcat([alpha, beta])
u = SX([0,0])



dt = 0.05
### DEFINE TRANSITION FUNCTION
def transition_model(x, u):
    f = vcat([x[0]+x[1]*dt, x[1], x[2]])
    return f

f = transition_model(x_sym, u_sym)


### DEFINE MEASUREMENT FUNCTION
def measurement_model(x):
    h = (x[0]**2 + x[2]**2) ** 0.5
    return h

h = measurement_model(x_sym)



### STATE COVARIANCE
P = SX_eye(3)
P *= 50

### PROCESS COVARIANCE
M = SX(3,3)
M[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
M[2,2] = 0.1

### MEASUREMENT COVARIANCE (NOISE)
R = np.diag([25])


### PLUG INTO EKF
ekf = ekf2.ExtendedKalmanFilter(state_vars=3, output_vars=1, control_vars=0, x0=x0, x_sym=x_sym, u=u, u_sym=u_sym, P=P, f=f, M=M, h=h, R=R)


### LOOP THROUGH ekf.update(z) and ekf.predict, appending ekf.x to an array
loops = 1000
t_time, xs = [], []
xs.append(x0)
for i in range(loops):
    t1 = time.time()
    
    z = 'your measurement matrix here'
    
    # ekf.update(z)
    xs.append(ekf.update(z)[0])
    ekf.predict()
    
    t2 = time.time()
    t_time.append(t2-t1)
    
    print(f"predict:\n {ekf.x}")
    
print("total run time:\n",sum(t_time))

