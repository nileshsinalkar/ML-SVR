# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:17:56 2019

@author: NSinalkar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
y=np.reshape(y,(-1, 1))

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)

#we cannot apply StandardScaler to a 1D array , but we can use scale
#from sklearn.preprocessing import scale
#y = scale(y)
y=sc_y.fit_transform(y)


plt.scatter(x,y,color='red')
plt.plot(x,y,color='red')
plt.show()

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)



plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.show()