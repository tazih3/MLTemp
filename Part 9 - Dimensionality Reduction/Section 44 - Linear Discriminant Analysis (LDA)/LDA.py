#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#LDA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/PCA/Wine.csv')
X=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

#SPLITTING INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#APPLYING LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train=lda.fit_transform(X_train,y_train) #Supervised model : we need to include the values of y
X_test=lda.transform(X_test) #no need for y_test because we're not fitting

#FITTING THE LOGISTIC REGRESSION MODEL TO THE DATASET
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#PREDICTING THE TEST SET RESULTS
y_pred=classifier.predict(X_test)

#EVALUATING THE PERFORMANCE OF THE CLASSIFIER: CONFUSION MATRIX
from sklearn.metrics import confusion_matrix #that's a function and not a class: no capital letter
cm=confusion_matrix(y_test,y_pred)

#VISUALISING THE TRAINING SET RESULTS:
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


#VISUALISING THE TEST SET RESULTS:
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
