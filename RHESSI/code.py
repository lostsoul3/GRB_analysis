import pandas as pd
import numpy as np
import math
from sklearn.mixture import GMM
from matplotlib import pyplot as plt
import itertools
from scipy import linalg
import matplotlib as mpl
from matplotlib.colors import LogNorm
from sklearn import mixture


df = pd.read_csv('data.csv',delimiter='|', header=None)
arr = df.as_matrix()

t90=[]
hr=[]

for i in range(len(arr)):
    arr[i][2] = math.log(arr[i][2])
    t90.append(arr[i][2])
    hr.append(arr[i][6])
    
data = np.column_stack((t90,hr))
data = data[~np.isnan(data).any(axis=1)]
data = data[np.isfinite(data).all(axis=1)]
N = np.arange(1,6)
data1 = np.vstack((t90,hr)).T
X=data
#fitting the model
def compute_GMM(N, covariance_type='full', n_iter=1000):
    models = [None for n in N]
    for i in range(len(N)):
        #print N[i]
        models[i] = GMM(n_components=N[i], n_iter=n_iter,
                        covariance_type=covariance_type)
        models[i].fit(X)
    return models

models = compute_GMM(N)

AIC = [m.aic(X) for m in models]
BIC = [m.bic(X) for m in models]

i_best = np.argmin(BIC)
gmm_best = models[i_best]
#print "best fit converged:", gmm_best.converged_
#print "BIC: n_components =  %i" % N[i_best]


color_iter = itertools.cycle(['c', 'gold', 'r', 'darkorange',
                              'c'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot()
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color,lw=2)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

plot_results(X, gmm_best.predict(X), gmm_best.means_, gmm_best.covars_, 0,
             'Gaussian Mixture')
plt.xlabel('Log(T90)', size=20)
plt.ylabel('Log(Hardness Ratio)', size=20)
"""
plt.plot(N, AIC, '-k', label='AIC',color ='blue', marker='o',lw=2)
plt.plot(N, BIC, ':k', label='BIC', color = 'red', marker='^',lw=2)
plt.legend(loc=1)
plt.xlabel('Number of components',size=20 )
plt.ylabel('AIC/BIC', size=20)
plt.xlim([0.95,5.05])"""

plt.show()

