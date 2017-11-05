import pandas as pd
import numpy as np

df = pd.read_csv('cplalpha.csv',delimiter='|', header=None)
arr = df.as_matrix()
df1 = pd.read_csv('splalpha.csv',delimiter='|', header=None)
arr1 = df.as_matrix()
df2 = pd.read_csv('pl.csv',delimiter='|', header=None)
arr2 = df2.as_matrix()
for i in range(len(arr2)):
    arr2[i][2]=arr2[i][2].strip()
    
alpha=[]
epeak=[]
enorm=[]

for i in range(len(arr)):
     if arr2[i][2]=='CPL':
         alpha.append(arr[i][2])
         epeak.append(arr[i][5])
         enorm.append(arr[i][8])
     else:
         alpha.append(arr1[i][2])
         epeak.append(np.nan)
         enorm.append(arr1[i][5])

for i in range(len(alpha)):
    alpha[i]=alpha[i].strip()
    if alpha[i]=='N/A':
        alpha[i]=np.nan
    else:
        alpha[i]=float(alpha[i])
        
for i in range(len(epeak)):
    epeak[i]=float(epeak[i])
for i in range(len(enorm)):
    enorm[i]=enorm[i].strip()
    if enorm[i]=='N/A':
        enorm[i]=np.nan
    else:
        enorm[i]=float(enorm[i])
