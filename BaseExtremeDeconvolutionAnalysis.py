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

import xdgmm
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

class BaseExtemeDeconvolutionAnalysis:

    def __init__(self, FILE_PATH, col_t90_idx, col_t90_err_dx, delimiter, col_hr_fluence_50_100, col_hr_fluence_20_50, col_hr_err):
        self.file_path = FILE_PATH
        self.col_t90 = col_t90_idx
        self.col_t90_error = col_t90_err_dx
        self.delimiter = delimiter
        self.col_hr_fluence_50_100 = col_hr_fluence_50_100
        self.col_hr_fluence_20_50 = col_hr_fluence_20_50
        self.col_hr_err = col_hr_err


    def get_valid_data(self, value):
        if math.isinf(value):
            return None
        else:
            return value

    def extract_data_and_data_error(self):
        batse_data = np.genfromtxt(self.file_path, delimiter=self.delimiter)
        self.t90 = []
        self.hardness_ratio = []
        self.dt90 = []
        self.dhr = []
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

'''
batse.txt
Batse data shape is (1973, 20)
HRr is 1934 and T90 is 1934
Delta HRr is 1934 and Delta T90 is 1934
[[ 3.08168728  0.22178302]
 [-0.2917376   0.52172023]]
Variance is [[[ 0.80785102  0.00256417]
  [ 0.00256417  0.15655426]]

 [[ 1.65568657 -0.14936446]
  [-0.14936446  0.48620792]]]
'''
def plot_ellipses(xdgmm):
    print(xdgmm.mu)
    print("Variance is {}".format(xdgmm.V))
    plot_results(X, xdgmm.predict(X, Xerr), xdgmm.mu, xdgmm.V, 0,
                 'Gaussian Mixture')
    plt.xlabel('Log(T90)', size=20)
    plt.ylabel('Log(Hardness Ratio)', size=20)
    plt.show()

def get_delta_aic_and_bic(X, Xerr, aic, bic, optimal_n_components):
    xdgmm = None
    xdgmm_2 = XDGMM(n_components=2, n_iter=1000)
    xdgmm_2.fit(X, Xerr)

    xdgmm_3 = XDGMM(n_components=3, n_iter=1000)
    xdgmm_3.fit(X, Xerr)

    # AIC(n_components=2) - AIC(n_components=3)
    delta_aic = aic[1] - aic[2]
    # BIC(n_components=2) - BIC(n_components=3)
    delta_bic = bic[1] - bic[2]

    print("Delta AIC is : {} and Delta BIC is : {}".format(delta_aic, delta_bic))
    means = None
    covars = None
    weights = None

    means = xdgmm_2.mu
    covars = xdgmm_2.V
    weights = xdgmm_2.weights
    xdgmm = xdgmm_2
    print("Means of 2 components are {} and Covars are {}".format(means, covars))

    print("Means of 3 components are {} and Covars are {}".format(xdgmm_3.mu, xdgmm_3.V))
    if optimal_n_components == 3:
        means = xdgmm_3.mu
        covars = xdgmm_3.V
        weights = xdgmm_3.weights
        xdgmm = xdgmm_3

    print("Total number of GRBs is {}".format(len(X)))
    i = 0
    for weight in weights:
        print("Group {} has {} datapoints\n".format(i,(weight * len(X))))
        i = i+1

    return delta_aic, delta_bic, xdgmm, weights, means, covars

X, Xerr = extract_data_and_data_error()
bic, aic, optimal_n_components = get_computed_models(X, Xerr)
plot_data_points(bic=bic, aic=aic)
delta_aic, delta_bic, xdgmm_final, weights, means, covars = get_delta_aic_and_bic(X, Xerr, aic, bic, optimal_n_components)
plot_ellipses(xdgmm_final)
