import numpy as np
import casadi as ca
from casadi import SX, MX, jacobian, sqrt, sin, cos, vcat, hcat, substitute, Function
import sympy as sp


# SYM VARIABLES
xs = SX.sym('x')
ys = SX.sym('y')
alphas = SX.sym('alpha')
betas = SX.sym('beta')
thetas = SX.sym('theta')


""" 
# STATE MEAN VECTOR
x = ca.vcat([xs, ys, thetas])

print("x:\n", x)

fxu = ca.vcat([xs-rs*sin(thetas) + rs*sin(thetas+betas),
              ys+rs*cos(thetas)- rs*cos(thetas+betas),
              thetas+betas])

print("fxu:\n", fxu)

F = jacobian(fxu,x)

print("F:\n",F)

Ft = F.T

print("Ft:\n", Ft)

"""

 
###USING THE Function FUNCTION IN casADi


# SYM VARIABLES
xs = SX.sym('x')
ys = SX.sym('y')
alphas = SX.sym('alpha')
betas = SX.sym('beta')


x = SX([3, 2])
x_sym = vcat([xs, ys])

u = SX([])
u_sym = vcat([])



f = vcat([xs, xs*2, xs*ys])


fxu = Function('f', [x_sym, u_sym], [f])


print("f result:", fxu(x,u))
print('r0:',fxu(x,u)[0])
print('q0:',fxu(x,u)[1])
print('t0:',fxu(x,u)[2])





""" 
### INV testing
import casadi as ca
import numpy as np

# Define a CasADi symbolic matrix
A = ca.SX([[1, 2], [3, 4]])
B = np.array([[1, 2], [3, 4]])

# Compute the inverse
A_inv = ca.inv(A)
B_inv = np.linalg.inv(B)

# Print the result
print(A_inv)
print(B_inv)




### SX_eye testing
print(ca.SX_eye(4))
 """
 
 