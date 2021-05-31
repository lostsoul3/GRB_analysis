'''
Here are the relevant columns you need to know
Col 2 T90
Col 3 error in T90
Col 4 fluence in 20-50 keV (F1)
Col  5 error in fluence in 20- 50 keV  (eF1)
Col 6  fluence in 50- 100 keV (F2)
Col 7  error in  fluence in 50- 100 keV (eF2)

For T90 use Col 2
So hardness ratio for BATSE will be F2/F1 or Col 6/Col 4
and ln(hardness) would  ln (col6/col4)
For our analysis since we only need the error in ln(hardness) we can do error propagation to calculate this
error in ln(hardness) = Error [ ln(f2) -ln(F1)]
                                 = sqrt [ (eF2/F2)^2 + (eF1/F1)^2]

'''
import os.path
import sys

import matplotlib
import pandas as pd
import numpy as np
import math
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import itertools
from scipy import linalg
from matplotlib.patches import Ellipse
import matplotlib as mpl
from matplotlib.colors import LogNorm
from sklearn import mixture

from xdgmm import XDGMM

FILE_PATH = 'batse.txt'
print(FILE_PATH)

# print(batse_data)
# print(batse_data.shape)
# exit(0)



def get_valid_data(value):
    if math.isinf(value):
        return None
    else:
        return value

def extract_data_and_data_error():
    batse_data = np.genfromtxt(FILE_PATH, delimiter=' ')
    t90 = []
    hardness_ratio = []
    dt90 = []
    dhr = []
    print("Batse data shape is {}".format(batse_data.shape))
    for i in range(len(batse_data)):
        # print(i)
        if batse_data[i][1] != float(0) and batse_data[i][3] != float(0) and batse_data[i][5] != float(0):
            hr = get_valid_data(math.log(float(batse_data[i][5] / batse_data[i][3])))
            t90_i = get_valid_data(math.log(batse_data[i][1]))

            if hr is not None and t90_i is not None:
                dt90_i = get_valid_data(batse_data[i][2])
                b = batse_data[i][6] / batse_data[i][5] ** 2 + (batse_data[i][4] / batse_data[i][3]) ** 2
                dhr_i = get_valid_data(math.sqrt(b))
                if dt90_i is not None and dhr_i is not None:
                    dt90.append(dt90_i)
                    dhr.append(dhr_i)
                    hardness_ratio.append(hr)
                    t90.append(t90_i)
    print("HRr is {} and T90 is {}".format(len(hardness_ratio), len(t90)))
    print("Delta HRr is {} and Delta T90 is {}".format(len(dhr), len(dt90)))

    mat_dt90 = np.zeros(len(dt90))
    mat_dhr = np.zeros(len(dhr))
    for i in range(len(dhr)):
        mat_dt90[i] = dt90[i]
        mat_dhr[i] = dhr[i]
    X = np.column_stack((t90, hardness_ratio))
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([mat_dt90 ** 2, mat_dhr ** 2]).T

    return X, Xerr


def get_computed_models(X, Xerr):
    param_range = np.arange(1, 6)
    n_iter = 10 ** 3
    xdgmm = XDGMM(n_iter=n_iter)
    bic, optimal_n_comp, lowest_bic = xdgmm.bic_test(X, Xerr, param_range)
    aic, optimal_n_aic_comp, lowest_aic = xdgmm.aic_test(X, Xerr, param_range)
    print("optimal bic {}".format(optimal_n_comp))
    print("optimal aic {}".format(optimal_n_aic_comp))
    return bic, aic, optimal_n_comp

def plot_data_points(bic, aic):
    N = np.arange(1, 6)
    plt.plot(N, aic, '-k', label='AIC', color='blue', marker='o', lw=2)
    plt.plot(N, bic, ':k', label='BIC', color='red', marker='^', lw=2)
    plt.legend(loc=1)
    plt.xlabel('Number of components', size=20)
    plt.ylabel('AIC/BIC', size=20)
    plt.xlim([0.95, 5.05])
    plt.show()




def plot_results(X, Y_, means, covariances, index, title):
    color_iter = itertools.cycle(['c', 'gold', 'r', 'darkorange',
                                  'c'])
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
        ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color,lw=2)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

def plot_ellipses(X, Xerr, optimal_n_components):
    xdgmm = XDGMM(n_components=2, n_iter=1000)
    xdgmm.fit(X, Xerr)
    plot_results(X, xdgmm.predict(X, Xerr), xdgmm.mu, xdgmm.V, 0,
                 'Gaussian Mixture')
    plt.xlabel('Log(T90)', size=20)
    plt.ylabel('Log(Hardness Ratio)', size=20)
    plt.show()

X, Xerr = extract_data_and_data_error()
# bic, aic, optimal_n_components = get_computed_models(X, Xerr)
# plot_data_points(bic=bic, aic=aic)
plot_ellipses(X, Xerr, 2)
