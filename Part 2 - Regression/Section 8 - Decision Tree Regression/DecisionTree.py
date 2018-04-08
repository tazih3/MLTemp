#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 01:53:51 2018

@author: hamzatazi
"""

#Decision Tree Regression Model
#Very interesting in many dimensions but not in 1 dimension !!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression/Decision_Tree_Regression/Position_Salaries.csv')
X=dataset.iloc[:,1:2].values 
y=dataset.iloc[:,2].values

#SPLITTING INTO TRAINING AND TEST SETS
"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

#FEATURE SCALING
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)"""

#FITTING THE DECISION TREE REGRESTTION TO THE DATASET
from sklearn.tree import DecisionTreeRegressor
regresor=DecisionTreeRegressor(random_state=0)
regresor.fit(X,y)


#PREDICTING A NEW RESULT
y_pred=regresor.predict(6.5)

#VISUALISING THE DECISION TREE REGRESSION RESULTS (WITH PRECISION)
X_grid=np.linspace(min(X),max(X),1000) #To have more precision
X_grid=X_grid.reshape((len(X_grid),1)) #to have a vector
plt.scatter(X,y,color='red') #The real observations
plt.plot(X_grid,regresor.predict(X_grid),color='blue') #On plot les x et prediction associée
plt.title("Truth or Bluff (Decision Tree)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot()
"""Obligé de mettre une plus grande résolution car modèle discontinu"""
#OBTENIR LE SCORE D'UNE RÉGRESSION:
"""regresor.score(X,y)"""

#PREDICTING NEW RESULTS WITH LR AND PR
y_pred=lin_reg.predict(X)
y_polypred=lin_reg2.predict(poly_reg.fit_transform(X))
print(y_pred,y_polypred)