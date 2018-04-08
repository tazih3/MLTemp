#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:00:01 2018

@author: hamzatazi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv('')
X=dataset.iloc[:,:].values
y=dataset.iloc[:,2].values

#SPLITTING INTO TRAINING AND TEST SETS
"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

#FITTING LR TO DATASET:
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#FITTING POLYNOMIAL REGRESSION TO THE DATASET
#Transforming X into X_poly
"""from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X) """ #Will create new matrix with X1^2... We have to fit our 
                                     #object to X and then transform it to X_poly
                                
lin_reg2=LinearRegression()
"""lin_reg2.fit(X_poly,y)"""

#Visualising the results:
X_grid=np.linspace(min(X),max(X),100) #To have more precision for the curve
X_grid=X_grid.reshape((len(X_grid),1)) #to have a vector
plt.scatter(X,y,color='red') #The real observations
plt.plot(X_grid,lin_reg.predict(X_grid),color='blue')
"""plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='orange')""" #On plot les X et la prediction associée
plt.title("Results of Linear and Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot()
"""print(lin_reg2.score(X_poly,y))"""

#PREDICTING NEW RESULTS WITH LR AND PR
y_pred=lin_reg.predict(X)
"""y_polypred=lin_reg2.predict(poly_reg.fit_transform(X))
print(y_pred,y_polypred)"""