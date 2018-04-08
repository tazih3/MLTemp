#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:09:30 2018

@author: hamzatazi
"""

"""IMPORTING THE LIBRARIES"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""IMPORTING THE DATASET"""
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
#dataset.describe() to have more informations
#dataset[['','','']] to get some columns



"""TAKING CARE OF MISSING DATA:"""
print(dataset.isnull().sum()) #detect which cells have missing values, and then count how many there are in each column with the command:
#Imputation (replacing a missing value with another value):
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

"""DROP COLUMNS WITH MISSING VALUES"""
#Columns with missing data
data_without_missing_values = original_data.dropna(axis=1)

#For test and training sets
cols_with_missing = [col for col in original_data.columns if original_data[col].isnull().any()] #sees if there
#is any missing data in this column
redued_original_data = original_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing, axis=1)


"""ENCODING CATEGORICAL DATA"""
#Easier way to doing it:
X=pd.get_dummies(X)

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


"""SPLITTING INTO TRAINING/TEST SETS:"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



"""FEATURE SCALING:"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

""" from sklearn.metrics import mean_absolute_error(ou autre)"""
