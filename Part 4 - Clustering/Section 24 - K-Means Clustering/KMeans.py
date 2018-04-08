#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:28:35 2018

@author: hamzatazi
"""
#%reset -f     pour enlever les variables    %clear  pour clear the console
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the mall dataset with pandas:
dataset=pd.read_csv('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/K_Means/Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of clusters:
from sklearn.cluster import KMeans

#Plotting the elbow method graph
#Computing the Within Cluster Sum of Squares (WCSS) for 10 different numbers of clusters
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init = 'k-means++',max_iter=300,n_init=10,random_state=0) 
    #Fitting KMeans to our data X + 'k-means++' to avoid the random intialization trap that would false our results
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the mall dataset
kmeans=KMeans(n_clusters=5, init='k-means++',max_iter=300,n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualizing the clusters (only in 2D or with dimensionality reduction)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careful') # rows that are in the first cluster, x= 1st column of data X (age here),y=2nd column of X
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()