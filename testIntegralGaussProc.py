print(__doc__)

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import matplotlib
matplotlib.use("wx")
from pylab import *
from math import *

np.random.seed(8)


L = 1
x0 = 0.5
k = 10
def f(t):
  return L/(1 + exp(-k*(t-x0)))

def df(x):
  """ differential of f w.r.t. t"""
  fx = L*k*exp(-k*(x-x0))/(1+exp(-k*(x-x0)))**2 
  #print fx
  return fx

#a = -0.5;
#b = 5;
#c = 0.5; 
#def df(x):
#    """The function to predict."""
#    return a*x**2+b*x+c

# now the noisy case
timeShift = 0.0
timePts = np.linspace(0+timeShift, 2*x0-timeShift, 100)

#X = np.linspace(0.1, L, 20)
X = np.array(map(f, timePts))
X = np.atleast_2d(X).T

# Observations and noise

y = np.array(map(df, timePts)).ravel()
dy = 0.2 + 0.2 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, L, 1000)).T

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='squared_exponential', theta0=1e+0,
                     #thetaL=1e-3, thetaU=1,
                     nugget=1e-1,#*(dy / y) ** 2,
                     random_start=100)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

x_true = np.array(map(f,timePts))
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = pl.figure()
dXdT_true = np.array(map(df,timePts))
print "dXdT_true", dXdT_true
pl.plot(x_true, dXdT_true, 'g', label=u'$df(x)$', linewidth=5.0)
pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$dx/dt$')
pl.ylim(-0.5, 4)
pl.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2)

savefig('figures/test/diffEqPlot.png')


def integrateTraj(xs,dXdT_pred):
  dXdT_pred = np.matrix(dXdT_pred)
  xs = np.matrix(xs)
  indices = np.array(range(len(xs)))
  dXs = (xs[indices] - xs[indices - 1])
  dXdivdXdT = np.divide(dXs, dXdT_pred[indices])
  t = np.cumsum(dXdivdXdT).T
  print "integrateTraj", xs.shape, dXs.shape, dXdT_pred.shape, dXdivdXdT.shape, t.shape
  #print t

  return t


y_pred = np.matrix(y_pred).T
t = integrateTraj(x, y_pred)
t = t - t[0] + timeShift # make the trajectory start from zero

x_true = np.matrix(x_true).T
dXdT_true = np.matrix(dXdT_true).T
tInteg_true = integrateTraj(x_true, dXdT_true)
tInteg_true = tInteg_true - tInteg_true[0] + timeShift

print t.shape, x.shape, tInteg_true.shape, x_true.shape, dXdT_true.shape

fig2 = pl.figure()
#print t, x
print tInteg_true
pl.plot(t, x, 'b', label='predicted trajectory')
pl.plot(timePts, x_true, 'g', label='true trajectory')
#pl.plot(tInteg_true, x_true, 'r', label='true integrated traj')
pl.legend(loc='upper left')
print "min max", np.min(x), np.max(x)
pl.ylim(np.min(x), np.max(x))
pl.xlim(0, 1)

savefig('figures/test/integTraj.png')
pl.show()
