#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:06:52 2018

@author: hamzatazi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv('')
X=dataset.iloc[:,1:2].values 
y=dataset.iloc[:,2].values

#SPLITTING INTO TRAINING AND TEST SETS
"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

#FEATURE SCALING: If we want an accurate prediction
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)"""

#FITTING THE REGRESSION MODEL TO THE DATASET
#Create your regresor here


#PREDICTING A NEW RESULT
y_pred=regresor.predict(X)

#VISUALISING THE REGRESSION RESULTS (WITH PRECISION)
X_grid=np.linspace(min(X),max(X),100) #To have more precision
X_grid=X_grid.reshape((len(X_grid),1)) #to have a vector
plt.scatter(X,y,color='red') #The real observations
plt.plot(X_grid,regresor.predict(X_grid),color='blue') #On plot les x et prediction associée
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.plot()

#OBTENIR LE SCORE D'UNE RÉGRESSION:
"""regresor.score(X,y)"""

#PREDICTING NEW RESULTS WITH LR AND PR
y_pred=lin_reg.predict(X)
y_polypred=lin_reg2.predict(poly_reg.fit_transform(X))
print(y_pred,y_polypred)