#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:32:26 2018

@author: hamzatazi
"""

"""Building a multivariable linar regression model
Always verify the 5 hypothesis:
    -Linearity
    -Homoscedasticity
    -Multivariate normality
    -Independence of errors
    -Lack of multicollinearity"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv')
X=dataset.iloc[:,:-1].values #we cannot see "objects" type
y=dataset.iloc[:,4].values

# ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder #this is a class
labelencoder_X=LabelEncoder() #this is an object of the class
X[:,3]=labelencoder_X.fit_transform(X[:,3]) #On met la colonne à laquelle on veut l'appliquer dans le fit_transform
                                            #Et on fait en sorte que la colonne qu'on veut prenne cette valeur
#Problem: Encoding like this might let the machine think that one country is greater than the other
#Solution: instead of encoding everything in one column, we separate the column in 3 other
#DummyVariables

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[3]) #On spécifie la colonne qu'on veut catégoriser
X=onehotencoder.fit_transform(X).toarray() #No need to specify the column because already done before

#Avoiding the Dummy Variables Trap
X=X[:,1:]

#SPLITTING INTO TRAINING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

"""A retenir : quand on crée n dummy variables, on n'en retient que n-1 dans notre modèle"""
"""The smaller the p-value, the greater the evidence against H0"""

#FITTING MLR TOT THE TRAINING SET
from sklearn.linear_model import LinearRegression
regresor=LinearRegression()
regresor.fit(X_train,y_train)

#PREDICTING THE TEST SET RESULTS
y_pred=regresor.predict(X_test)

#BUILDING THE OPTIMAL MODEL USING MANUAL BACKWARD ELIMINATION
import statsmodels.formula.api as sm
"""We need to add the intercept for the constant (y=b0+b1*x1...)"""
X=np.append(np.ones((50,1)),X,1) """axis=1 pour ajouter une colonne, axis=0 pour une ligne
#On commence par le np.ones pour l'avoir au tout début comme intercept"""

X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit() #Fitting the model with all possible predictors
regressor_OLS.summary() #the lower the p-value, the more you keep that variable
#Parce que c'est p-value>alpha alors on rejette H0 qui est que la variable est dans le modèle
#On retire donc au fur et à mesure les plus grandes p-value

X_opt=X[:,[0,1,3,4,5]] #x2 removed
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]] #x1 removed
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]] #x4 removed
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

#Choice to make : x3 is the best variable to keep, but x5 is good too because the p-value
#is only 1% above our treshhold of 5% so I think we should keep both.

X_opt=X[:,[0,3]] #x5 removed
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

##########################################################################################
#AUTOMATIC BACKWARD ELIMINATION WITH P-VALUES ONLY
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#AUTOMATIC BACKWARD ELIMINATION WITH P-VALUES AND ADJUSTED R-SQUARED
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
