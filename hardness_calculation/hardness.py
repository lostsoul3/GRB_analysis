import numpy as np
import math

#-- Data generated from datagenerator.py
data = np.genfromtxt('data_file.csv',delimiter=',')


#--Power law functions:
def func(E,i):
    return E**(data[i][0])

def func2(E,i):
    return ((E**(data[i][0]))*np.exp(-E/data[i][1]))


#--Calculation for Hardness    
hardness = np.zeros(375)
dE = 0.0001
j=0
for i in range(0,375):
    E1 = np.arange(50,100,dE)*(1+data[i][2])
    E2 = np.arange(25,50,dE)*(1+data[i][2])
    if (math.isnan(data[i][1])==1):
        hardness[j] = (np.sum(func(E1,i)*E1))/(np.sum(func(E2,i)*E2))
        j+=1
    else:
        hardness[j] = (np.sum(func2(E1,i)*E1))/(np.sum(func2(E2,i)*E2))
        j+=1


#--Result array with alpha, epeak(wherever applicable), redshift, trigger number, hardness
df = np.zeros((375,5))
for i in range(len(df)):
    df[i] = [data[i][0],data[i][1],data[i][2],data[i][3],hardness[i]]


#--Make final datafile
np.savetxt('hardness_data.csv',df,delimiter=',',header = 'Alpha, Epeak, Redshift, Trigger Number, Hardness')

