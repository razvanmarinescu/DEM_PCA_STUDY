import numpy as np
from numpy import ndarray
from sklearn import gaussian_process

def f(x):
  return x * np.sin(x)

X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
Y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X, Y)  
#GaussianProcess(beta0=None, corr=<function squared_exponential at 0x...>,
#        normalize=True, nugget=array(2.22...-15),
#        optimizer='fmin_cobyla', random_start=1, random_state=...
#        regr=<function constant at 0x...>, storage_mode='full',
#        theta0=array([[ 0.01]]), thetaL=array([[ 0.0001]]),
#        thetaU=array([[ 0.1]]), verbose=False)

y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)


import matplotlib.pyplot as plt
print X.shape, Y.shape, x.shape, y_pred.shape, sigma2_pred.shape
#print X
#print Y
#print x
#print y_pred
plt.plot(X, Y, 'ro', x, y_pred, 'b')
plt.ylabel('dx/dt')
plt.show()
