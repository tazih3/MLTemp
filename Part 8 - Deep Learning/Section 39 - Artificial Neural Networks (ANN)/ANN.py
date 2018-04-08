#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 00:25:06 2018

@author: hamzatazi
"""
#ARTIFICIAL NEURAL NETWORKS:
"""Here, it is a classification problem"""
# Data Preprocessing:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding the categorical variables (ONLY THE INDEPENDENT HERE BECAUSE y IS 0/1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in [1,2]:
    labelencoder_Xi = LabelEncoder()
    X[:, i] = labelencoder_Xi.fit_transform(X[:, i])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:13] #Avoiding the dummy variable trap and deleting one dummy variable (for the 3 countries)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling: necessary for ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making the ANN:
import keras
from keras.models import Sequential
from keras.layers import Dense

#Adding the input layer and the first hidden layer:
"""We'll choose rectifier max(x,0) function for hidden layers, sigmoid for output"""
classifier=Sequential() #Intializing the ANN
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11)) #le init c'est pour initialiser les poids par loi uniforme proche de 0
"""Number of hidden layers : average between number of independent variables and dependent variables (input and output layer)"""

#Add a hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu')) #no need to specify input_dim here because already known

#Add the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid')) #no need to specify input_dim here because already known
"""use softmax if the dependent variable has more than one category (and change units)"""

#Compiling the ANN: i.e. applying stochastic gradient descent
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #'adam' est un stochastic gradient descent algorithm efficace
"""if dependent variable more than 2 categories : categorical_crossentropy """

#Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Predicting the Test set results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5) #avoir des booléens car on a des proba ici avec la sigmoide

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)