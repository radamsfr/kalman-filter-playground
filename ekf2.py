# import numpy as np
from casadi import SX, MX, jacobian, sqrt, sin, cos, arctan2, vcat, hcat, Function, inv, SX_eye, evalf


class ExtendedKalmanFilter():
    def __init__(self, state_vars, output_vars, control_vars, x0, x_sym, u, u_sym, P, f, M, h, R):
        self.state_vars = state_vars
        self.output_vars = output_vars
        self.control_vars = control_vars
        
        
        """ 
        MATRICES THAT USE SYMBOLIC VARIABLES MUST USE vcat() MATRIX,
        OTHERWISE USE SX MATRIX. (DO NOT USE MX)                       
        """
        
        self.x = x0  # STATE MEAN 
        self.x_sym = x_sym  # STATE MEAN (SYMBOLIC VARAIABLES)
        self.P = P  # STATE COVARIANCE
    
        self.f = f  # STATE TRANSITION FUNCTION (SYMBOLIC VARIABLES)
        self.M = M  # PROCESS COVARIANCE/NOISE (aka Q)
        
        self.u = u  # CONTROL INPUT
        self.u_sym = u_sym  # CONTROL INPUT (SYMBOLIC VARIABLES)
        
        self.z = SX(state_vars, 1)  # MEASUREMENT MEAN
        self.R = R # MEASUREMENT COVARIANCE
        
        self.h = h # MEASUREMENT TRANSITION FUNCTION (SYMBOLIC VARIABLES)
        
        
        # FUNCTIONS FROM PROCESS AND MEASUREMENT MATRICES
        self.fxu = Function('fxu', [self.x_sym, self.u_sym], [self.f])  
        
        self.hx = Function('hx', [self.x_sym], [self.h])
        
        # JACOBIANS AND CORRESPONDING FUNCTIONS
        self.F = jacobian(self.f, self.x_sym)
        self.Fxu = Function('Fxu', [self.x_sym, self.u_sym], [self.F])  
        
        self.H = jacobian(self.h, self.x_sym)
        self.Hx = Function('Hx', [self.x_sym], [self.H])
        
        self.V = jacobian(self.f, self.u_sym)
        self.Vxu = Function('Vxu', [self.x_sym, self.u_sym], [self.V]) 
        
        # MAKE SURE V CAN MATRIX MULTIPLY WITH M
        # print("init V:\n", self.V)


        
    def update(self, z):
        self.z = z  # MEASUREMENT MEAN (MUST BE MATRIX (SX))

        self.y = self.z - self.hx(self.x)  # PLUG IN VALUES OF STATE MEAN x INTO FUNCTION hx    
        
        self.H = self.Hx(self.x)
        Ht = self.H.T
        
        # print("H:\n", self.H)
        # print("Ht:\n", Ht)
        
        S = self.H @ self.P @ Ht + self.R
        self.K = self.P @ Ht @ inv(S)
        
        self.x = self.x + self.K @ self.y
        self.x=evalf(self.x)

        I = SX_eye(self.P.shape[0])  # Identity matrix
        self.P = (I - self.K @ self.H) @ self.P
        self.P = evalf(self.P)
        
        return (self.x, self.P)
        



    def predict(self):
        self.x = self.fxu(self.x, self.u)
        
        self.x=evalf(self.x)


        self.F = self.Fxu(self.x, self.u)
        Ft = self.F.T
        self.V = self.Vxu(self.x, self.u)
        Vt = self.V.T
        
        
        # print("F:\n", self.F)
        # print("Ft:\n", Ft)
        # print("P:\n", self.P)
        
        # print("V:\n", self.V)
        # print("Vt:\n", Vt)
        # print("M:\n", self.M)
        
        
        self.P = self.F @ self.P @ Ft + self.V @ self.M @ Vt
        
        self.P = evalf(self.P)
    
        
        return (self.x, self.P)