#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 00:57:41 2018

@author: hamzatazi
"""
#SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/SVR/Position_Salaries.csv')
X=dataset.iloc[:,1:2].values 
y=dataset.iloc[:, 2].values

#SPLITTING INTO TRAINING AND TEST SETS
"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=y.reshape(-1,1)
y=sc_y.fit_transform(y)
"""ici feature scaling obligatoire avec svr"""

#FITTING THE SVR MODEL TO THE DATASET
from sklearn.svm import SVR
regresor=SVR(kernel='rbf') #C to play with overfit #most important is kernel, rbf is gaussian
regresor.fit(X,y)

#PREDICTING A NEW RESULT
y_pred=sc_y.inverse_transform(regresor.predict(sc_X.transform(np.array([[6.5]])))) 
#array obligé (matrix meme pas vector) pour ça que [[]] et il faut transform and invert_transform

#VISUALISING THE SVR RESULTS (WITH PRECISION)
X_grid=np.linspace(min(X),max(X),100) #To have more precision
X_grid=X_grid.reshape((len(X_grid),1)) #to have a vector
plt.scatter(X,y,color='red') #The real observations
plt.plot(X_grid,regresor.predict(X_grid),color='blue') #On plot les x et prediction associée
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#OBTENIR LE SCORE D'UNE RÉGRESSION:
"""regresor.score(X,y)"""

"""#PREDICTING NEW RESULTS WITH LR AND PR
y_pred=lin_reg.predict(X)
y_polypred=lin_reg2.predict(poly_reg.fit_transform(X))
print(y_pred,y_polypred)"""