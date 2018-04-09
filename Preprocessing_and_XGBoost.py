#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:14:21 2018

@author: hamzatazi
"""
#IMPORTING THE DATASETS
import os
#print(os.listdir("../input"))
train_raw=pd.read_csv('../input/train.csv')
train=train_raw.copy(deep=True)
test=pd.read_csv('../input/test.csv')
data=[train,test]

#TAKING CARE OF MISSING DATA
print('Train columns with null values:\n', train.isnull().sum())
print("-"*30)
print('Test columns with null values:\n', test.isnull().sum())

#Replacing missing values
for x in data:
    x['Age'].fillna(x['Age'].mean(),inplace=True)
    x['Embarked'].fillna(x['Embarked'].mode()[0], inplace = True) #mode()[0] : first value of a series with most common value in the dataset
    x['Fare'].fillna(x['Fare'].mean(), inplace = True)
train.drop(['PassengerId','Cabin','Ticket'],axis=1,inplace=True)
test.drop(['Cabin','Ticket'],axis=1,inplace=True)

#Feature Engineering:
#Creating a new relevant feature in order to get rid of 2:
for x in data:
    x['FamilySize']=x['SibSp']+x['Parch']+1 #adding number of siblings, parents and 1 to count the person herself
    x['IsAlone']=1 #initializing to 1 (Yes)
    x['IsAlone'].loc[x['FamilySize']>1]=0 #changing it to 0 (No) if the corresponding family size is greater than 1
    x['Title'] = x['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] #The first str.split[1] takes the 2nd part in which there is only the family name, then the second splits
    #between the title and the rest (regarding the ".")
for x in data:
    x.drop(['Name','SibSp','Parch'],axis=1,inplace=True)
y_train=train.iloc[:,0].values
y_val=test.iloc[:,0].values

#Encoding the data:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
for x in data:
    x['Sex']=label.fit_transform(x['Sex'])
    x['Embarked']=label.fit_transform(x['Sex'])
    x['Title']=label.fit_transform(x['Title'])
X_train=train.iloc[:,1:].values
X_test=test.iloc[:,1:].values
onehotencoder = OneHotEncoder(categorical_features = [1]) #Encoding the Sex
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

onehotencoder1 = OneHotEncoder(categorical_features = [4]) #Encoding the Embarked
X_train = onehotencoder1.fit_transform(X_train).toarray()
X_test = onehotencoder1.fit_transform(X_test).toarray()

onehotencoder2 = OneHotEncoder(categorical_features = [7]) #Encoding the titles
X_train = onehotencoder2.fit_transform(X_train).toarray()
X_test = onehotencoder2.fit_transform(X_test).toarray()

# Splitting into training and test set for train data:
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

#Using XGBoost:
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred = classifier.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)
print('The accuracy on the training set is:',(cm[0][0]+cm[1][1])/(np.sum(cm)))