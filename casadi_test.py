import numpy as np
import casadi as ca
from casadi import SX, MX, jacobian, sqrt, sin, cos, vertcat, substitute
import sympy as sp

"""

# Create symbolic variables
x = ca.MX.sym('x')
y = ca.MX.sym('y')

# Create functions
f1 = x**2 + y
f2 = ca.sin(x) - ca.cos(y)
f3 = x*y
f4 = ca.exp(x/y)

# Create a matrix of functions
matrix_of_functions = ca.vertcat(
    ca.horzcat(f1, f2),
    ca.horzcat(f3, f4)
)

# Print the matrix
print(matrix_of_functions)

# Compute the jacobian
jacobian = ca.jacobian(matrix_of_functions, x)
print(jacobian)


# Create a CasADi function that takes x and y as inputs and returns the matrix of functions
func = ca.Function('func', [x, y], [matrix_of_functions])

# Define the values for x and y
x_val = 1.0
y_val = 2.0

# Evaluate the matrix of functions with the specific values
result = func(x_val, y_val)

# Print the result
print(result)

"""

xs = MX.sym('x')
ys = MX.sym('y')
thetas = MX.sym('theta')
alphas = MX.sym('alpha')
betas = MX.sym('beta')
rs = MX.sym('r')

pxs = MX.sym('p_x')
pys = MX.sym('p_y')

'''
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

'''

z = ca.vcat([sqrt((pxs-xs)**2 + (pys-ys)**2),
            ca.arctan2(pys-ys, pxs-xs) - thetas])

print("z:\n", z)

##### WRITE FUNCTIONS TO CONVERT casADi TO SymPy TO MAKE EQUATIONS READABLE
""" 
def casadi_to_sympy(expr):
    # Convert a CasADi expression to a SymPy expression.
    if isinstance(expr, ca.MX):
        # Handle CasADi symbols
        if expr.is_symbolic():
            return sp.symbols(str(expr))
        # Handle CasADi constants
        elif expr.is_constant():
            return sp.Float(float(expr))
        # Handle CasADi expressions
        else:
            # Convert CasADi expression to string and parse it with SymPy
            return sp.sympify(str(expr))
    else:
        raise TypeError("Unsupported type")

def casadi_matrix_to_sympy(matrix):
    # Convert a CasADi matrix to a SymPy matrix.
    rows, cols = matrix.shape
    sympy_matrix = sp.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            sympy_matrix[i, j] = casadi_to_sympy(matrix[i, j])
    return sympy_matrix

def pretty_print_casadi_matrix(matrix):
    # Pretty print a CasADi matrix using SymPy.
    sympy_matrix = casadi_matrix_to_sympy(matrix)
    sp.pprint(sympy_matrix)


print("PRINTING: ")
pretty_print_casadi_matrix(F) 

"""