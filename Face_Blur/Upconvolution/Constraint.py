import numpy as np
from scipy.optimize import minimize, LinearConstraint
#from scipy import optimize
#from scipy.optimize import NonlinearConstraint
import scipy as sp

# t = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
t = np.array([[1,2,255,4],[113,84,5,87],[9,110,171,212],[3,54,15,16]])
print(t)
# print("shape of t: ",t.shape)

padding_t = np.pad(t, (2, 2), mode='constant', constant_values=0)
print(padding_t)

f = np.array([[2,1,0],[0,1,2],[1,0,2]])
print(f)
print("shape of f: ",f.shape)

a = t.shape[0] - f.shape[0] + 1
b = t.shape[0] - f.shape[1] + 1

result = []

for rn in range(a):
    for cn in range(b):
        result1 = t[rn:rn+f.shape[0], cn:cn+f.shape[1]]*f
        result.append(np.sum(result1))

result = np.array(result).reshape(a, b)

print("shape of result")
print(result)

# def function(x, y):
#     return y[0]*x[0] + y[1]*x[1] + y[2]*x[2] + y[3]*x[3] + y[4]*x[4] + y[5]*x[5] + y[6]*x[6] + y[7]*x[7] + y[8]*x[8] - y[9]
def function(x):
    return 2*x[0] + 1*x[1] + 0*x[2] + 0*x[3] + 1*x[4] + 2*x[5] + 1*x[6] + 0*x[7] + 2*x[8] - 449

# 얘는 제약 조건
# def function(x, y):
#     return 2*x[0] + 1*x[1] + 0*x[2] + 0*x[3] + 1*x[4] + 2*x[5] + 1*x[6] + 0*x[7] + 2*x[8] - 449
# def eq_constraint(x):
#     return 2*x[0] + 1*x[1] + 0*x[2] + 0*x[3] + 1*x[4] + 2*x[5] + 1*x[6] + 0*x[7] + 2*x[8] - 449


# Init_Point = np.array([1.,2.,255.,113.,84.,5.,9.,110.,171.])
Init_Point = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
y = np.array([2, 1, 0, 0, 1, 2, 1, 0, 2, 449])

op = sp.optimize.fmin_slsqp(function, Init_Point, eqcons=[function])
# op = sp.optimize.fmin_slsqp(function, np.array([1.,2.,3.,5.,6.,7.,9.,10.,12.]), eqcons=[function])
# op = sp.optimize.fmin_slsqp(function, Init_Point, args=(Init_Point, y), eqcons=[function])
print(op)
