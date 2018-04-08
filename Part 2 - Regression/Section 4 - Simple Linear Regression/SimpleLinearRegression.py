#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:51:37 2018

@author: hamzatazi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression/Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#SPLITTING INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split #Cross_validation replacement
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#FEATURE SCALING
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #Calling the class, this is a function that returns an result of itself
regressor.fit(X_train,y_train) #The machine then learned the correlations

#PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

#VISUALISING THE TRAINING SET RESULTS
plt.scatter(X_train,y_train,color='red') #Nuage de points = Scatter Plot
plt.plot(X_train,regressor.predict(X_train),color='blue') #ATTENTION pas le test !!
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#VISUALISING THE TEST SET RESULTS
plt.scatter(X_test,y_test,color='orange')
plt.plot(X_train,regressor.predict(X_train))
#Etant donné que le modèle s'est entraîné sur le training set, pas 
#Besoin de changer le plt.plot car on construirait juste une nouvelle
#Regression : on garde donc un plot avec la régression sur X_train
#Et on voit comment ça se comporte par rapport aux points de y_test
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
