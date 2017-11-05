import numpy as np
import math


# Generate data:

cpl = np.genfromtxt('cplalpha.csv',delimiter = '|',usecols=(0,2,5), dtype=("|S24",float,float),autostrip = True)
spl = np.genfromtxt('splalpha.csv',delimiter='|',dtype = (np.dtype(str,12),float),usecols=(0,2),autostrip = True)
pl = np.genfromtxt('pl.csv',delimiter = '|',usecols=(0,2), dtype=(np.dtype(str,12),str),autostrip = True)
res = np.genfromtxt('real_redshift.csv',delimiter=',',usecols=(0,1),dtype=(np.dtype(str,12),float),autostrip =True)
pl1 = np.genfromtxt('pl.csv',delimiter = '|',usecols=(1),autostrip = True)


# Replaced 'N/A' with np.nan:

count =0
for i in range(len(spl)):
    if spl[i][1]=='N/A':
        count+=1
        spl[i][1] = np.nan
count


# Loop for compiling all data together:

data =np.zeros((len(pl),2))
df =np.zeros((len(pl),4))
# count,count2 = 0,0
for i in range(len(pl)):
    if pl[i][1]=='CPL':
        data[i][0],data[i][1] = cpl[i][1],cpl[i][2]
    else:
        data[i][0],data[i][1] = spl[i][1], np.nan
k=0
for i in range(len(pl)):
    for j in range(len(res)):
        if (pl[i][0]==res[j][0]):
            # count+=1
            df[k][0],df[k][1],df[k][2],df[k][3] = data[i][0],data[i][1],res[j][1],pl1[i]
            k+=1
        else:
            # count2+=1
#             data[i][2] = np.nan


# Write to file:

np.savetxt('data_file.csv',df,delimiter=',',header = '#Alpha,Epeak,Redshift,Trigger Number')
