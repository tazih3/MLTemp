#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:10:31 2018

@author: hamzatazi
"""

#The objective is to predict if a review is positive or negative
#NATURAL LANGUAGE PROCESSING:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset:
dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Natural_Language_Processing/Restaurant_Reviews.tsv',delimiter='\t',quoting=3) #We're ignoring the double quotes

#Cleaning the texts:
"""Tokenization: split the different reviews into different words which will be only relevant words
Thanks to text pre-processing then one column for each word and for each review each column
will contain the number of times the associated word appears in the review"""
import re #cleaning the texts
import nltk
nltk.download('stopwords') #On supprime les mots inutiles comme les prépositions etc
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(1000):
    review= re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #On ne garde que les lettres et on remplace par espace
    review=review.lower() #on enlève les majuscules
    review=review.split() #split the review into different words: becomes a list of different words
    ps=PorterStemmer() #on ne garde que la racine des mots importants : "love" au lieu de "loved" ou "loving"...
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
#Taking away all the words in this review that are not in this stopwards list
#Set permet à l'algo de parcourir plus vite qu'avec une liste
    review=' '.join(review,) #Go back from list to string
    corpus.append(review)
    
#BAG OF WORDS MODEL : FILTERING THE WORDS THAT DON'T APPEAR A LOT --> Minimizing # of words
"""Each word correspond to a column and each row to a review"""
"""We're gonna build a sparse matrix of features(with a lot of zeros)"""
"""We're gonna obtain a classic classification model through this model !!"""
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #max_features ne garde queles x relevant words (qui apparaissent plusieurs fois)
X=cv.fit_transform(corpus).toarray() #Gives the sparse matrix which represents the independent variables
y= dataset.iloc[:,1].values


"""To reduce sparsity 2 solutions : avoir un max_features + faible OU Dimensionality reduction"""
#Classification: NLP we use Random Forest or Naive Bayes (experience).
"""Here we're using Random Forest"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) #Not necessary because Decision Trees not based on Euclidian Distance"""

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=300, random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred=classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
precision=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])