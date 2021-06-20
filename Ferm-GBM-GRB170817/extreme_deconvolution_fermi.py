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

FILE_PATH = 'hrdata_10yr_T90_Final_w_err copy.txt'
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
    fermi_data = np.genfromtxt(FILE_PATH, delimiter='\t')
    t90=[]
    hr=[]
    dt90=[]
    dhr=[]
    COLUMN_T90_ERROR = 4
    COLUMN_T90 = 3
    COLUMN_HR_ERROR = 2
    COLUMN_HR = 1

    print("Batse data shape is {}".format(fermi_data.shape))

    for i in range(len(fermi_data)):
        if fermi_data[i][COLUMN_T90] > 0 and fermi_data[i][COLUMN_HR] > 0:
            t90.append(math.log(fermi_data[i][3]))
            hr.append(math.log(fermi_data[i][1]))
            dhr.append(fermi_data[i][COLUMN_HR_ERROR] / fermi_data[i][COLUMN_HR])
            dt90.append((fermi_data[i][COLUMN_T90_ERROR]/fermi_data[i][COLUMN_T90]))
    print(len(dt90))

    print("HRr is {} and T90 is {}".format(len(hr), len(t90)))
    print("Delta HRr is {} and Delta T90 is {}".format(len(dhr), len(dt90)))

    mat_dt90 = np.zeros(len(dt90))
    mat_dhr = np.zeros(len(dhr))
    for i in range(len(dhr)):
        mat_dt90[i] = dt90[i]
        mat_dhr[i] = dhr[i]
    X = np.column_stack((t90, hr))
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

'''
dell@dell-Inspiron-5559:~/Desktop/iit_hyd/GRB_analysis/Ferm-GBM-GRB170817$ python3 extreme_deconvolution_fermi.py 
hrdata_10yr_T90_Final_w_err copy.txt
Batse data shape is (2330, 5)
2329
HRr is 2329 and T90 is 2329
Delta HRr is 2329 and Delta T90 is 2329
N = 1 , BIC = 13406.952679253236
N = 2 , BIC = 13036.83947636553
N = 3 , BIC = 13039.010819651332
N = 4 , BIC = 13068.833222885023
/usr/local/lib/python3.8/dist-packages/scikit_learn-0.23.2-py3.8-linux-x86_64.egg/sklearn/mixture/_base.py:265: ConvergenceWarning: Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.
  warnings.warn('Initialization %d did not converge. '
N = 5 , BIC = 13107.397397757439
N = 1 , AIC = 13378.186707903815
N = 2 , AIC = 12973.55434533583
N = 3 , AIC = 12941.20769778889
N = 4 , AIC = 12936.531775161904
N = 5 , AIC = 12940.469467438155
optimal bic 2
optimal aic 4
extreme_deconvolution_fermi.py:102: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string "-k" (-> color='k'). The keyword argument will take precedence.
  plt.plot(N, aic, '-k', label='AIC', color='blue', marker='o', lw=2)
extreme_deconvolution_fermi.py:103: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string ":k" (-> color='k'). The keyword argument will take precedence.
  plt.plot(N, bic, ':k', label='BIC', color='red', marker='^', lw=2)
Delta AIC is : 32.346647546939494 and Delta BIC is : -2.171343285801413
Means of 2 components are [[-0.09604603  0.34184669]
 [ 3.27362629 -0.45537474]] and Covars are [[[ 0.96886125 -0.19369862]
  [-0.19369862  0.2349018 ]]

 [[ 1.10733344 -0.04387518]
  [-0.04387518  0.27915813]]]
Means of 3 components are [[ 3.71159734 -0.45779885]
 [-0.43234508  0.43813393]
 [ 2.59347773 -0.40661163]] and Covars are [[[ 0.75037429 -0.03604153]
  [-0.03604153  0.18740657]]

 [[ 0.63348061 -0.17304833]
  [-0.17304833  0.17443373]]

 [[ 1.29938524 -0.11672536]
  [-0.11672536  0.40480485]]]
Total number of GRBs is 2329
Group 0 has 358.95189637625407 datapoints

Group 1 has 1970.0481036237463 datapoints

[[-0.09604603  0.34184669]
 [ 3.27362629 -0.45537474]]
Variance is [[[ 0.96886125 -0.19369862]
  [-0.19369862  0.2349018 ]]

 [[ 1.10733344 -0.04387518]
  [-0.04387518  0.27915813]]]

'''
