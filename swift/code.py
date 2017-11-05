import numpy as np
import pandas as pd
from sklearn.mixture import GMM
from matplotlib import pyplot as plt
import math

#building dataset
df = pd.read_csv('amldata.csv', sep='\t', header=None)
arr = df.as_matrix()
hr=[]
for i in range(len(arr)):
    a = arr[i][5]+arr[i][6]
    b = arr[i][5]-arr[i][6]
    h1=float((a+b)/2)
    a = arr[i][3]+arr[i][4]
    b = arr[i][3]-arr[i][4]
    h2=float((a+b)/2)
    a = h1/h2
    a = math.log(a)
    hr.append(a)
    
t90=[]
for i in range(len(arr)):
    a = math.log(arr[i][0])
    t90.append(a)
    
data = np.column_stack((t90,hr))
data = data[~np.isnan(data).any(axis=1)]
data = data[np.isfinite(data).any(axis=1)]
N = np.arange(1,6)
data1 = np.vstack((t90,hr)).T

#fitting the model
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i]=GMM(N[i]).fit(data)
    
AIC = [m.aic(data) for m in models]
BIC = [m.bic(data) for m in models]
best_index = np.argmin(BIC)
M_best = models[best_index]
print "best fit converged:", M_best.converged_
print "BIC: n_components =  %i" % N[best_index]

#plot the graph
plt.plot(t90, hr, 'ro')
fig = plt.figure(figsize=(5, 1.66))
fig.subplots_adjust(wspace=0.45,
                    bottom=0.25, top=0.9,
                    left=0.1, right=0.97)
ax = fig.add_subplot(132)
ax.plot(N, AIC, '-k', label='AIC')
ax.plot(N, BIC, ':k', label='BIC')
ax.legend(loc=1)
ax.set_xlabel('N components')
plt.setp(ax.get_yticklabels(), fontsize=7)
plt.show()

