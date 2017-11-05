import pandas as pd
import numpy as np
import math
from sklearn.mixture import GMM
from matplotlib import pyplot as plt

df = pd.read_csv('pl.csv', sep='|', header=None)
arr = df.as_matrix()
cpl=[]
for i in range(len(arr)):
    arr[i][2]=arr[i][2].strip()
    if arr[i][2]=='CPL':
        arr[i][0]=arr[i][0].strip()
        cpl.append(arr[i][0])
        
df = pd.read_csv('cpl.csv', sep='|', header=None)
arr = df.as_matrix()
df1 = pd.read_csv('spl.csv', sep='|', header=None)
arr1 = df1.as_matrix()

for i in range(len(arr1)):
    arr1[i][5]=arr1[i][5].strip()
    arr1[i][8]=arr1[i][8].strip()
    if arr1[i][5]=='N/A' or arr1[i][8]=='N/A':
        arr1[i][5]=np.nan
        arr1[i][8]=np.nan

for i in range(len(arr1)):
    arr[i][5]=arr[i][5].strip()
    arr[i][8]=arr[i][8].strip()
    if arr[i][5]=='N/A' or arr[i][8]=='N/A':
        arr[i][5]=np.nan
        arr[i][8]=np.nan
           
j=0
ef1=[]
ef2=[]
for i in range(len(arr)):
   arr[i][0]=arr[i][0].strip()
   if arr[i][0]==cpl[j]:
       ef1.append(arr[i][5])
       ef2.append(arr[i][8])
       j=j+1
   else:
       ef1.append(arr1[i][5])
       ef2.append(arr1[i][8])

    
t90=[]
df = pd.read_csv('t90.csv', sep='|', header=None)
arr = df.as_matrix()
for i in range(len(arr)):
    arr[i][8]=arr[i][8].strip()
    if arr[i][8]==' N/A ' or arr[i][8]=='N/A':
        arr[i][8]=np.nan
    else:
        a=float(arr[i][8])
        arr[i][8]=a
    t90.append(arr[i][8])
