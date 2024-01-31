import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
import sys

#a = np.loadtxt('fx.txt', delimiter=',')
#b = np.loadtxt('gx.txt', delimiter=',')
fgw = np.loadtxt('fgw.txt', delimiter=',')
a = fgw[:,0]
b = fgw[:,1]
w = fgw[:,2]
del fgw

# Cost matrix
x = np.loadtxt('x.txt', delimiter=',')
M = np.sin(x[1,:]) * np.sin(np.reshape(x[1,:],(-1,1))) * np.cos(x[0,:]-np.reshape(x[0,:],(-1,1))) + np.cos(x[1,:])*np.cos(np.reshape(x[1,:],(-1,1)))
M[M>1] = 1
M[M<-1] = -1
M = np.real(np.arccos(M))**2

# Barycenter weights
if len(sys.argv) > 1:
  delta = float(sys.argv[1])
else:
  delta = 0.5# 0<=delta<=1

weights = np.array([delta, 1 - delta])

# creating matrix A containing all distributions
A = np.vstack((a, b)).T
del a, b

# Regularization parameter for Sinkhorn
if len(sys.argv) > 2:
  reg = float(sys.argv[2])
else:
  reg = 1e-3

# Number of iterations
if len(sys.argv) > 3:
  iter = int(sys.argv[3])
else:
  iter = 100

  
# function barycenter usually returns division by 0 error, so we try barycenter_debiased
if reg > 0:
  bary_wass = ot.bregman.barycenter(A, M, reg, weights, numItermax=iter, verbose=True)
elif reg == 0:
  bary_wass = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)

np.savetxt("bary_wass.txt", bary_wass, delimiter=',')

