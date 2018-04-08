#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 01:37:53 2018

@author: hamzatazi
"""

#APriori - Association Rule Learning
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/hamzatazi/Desktop/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python')

#Importing the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None) #Si pas de titre de colonne
transactions=[]
for i in range (0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)]) #Creating a list of list

#Training Apriori on the dataset:
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
"""Min_support is calculated as products purchased at least 3times a day i.e. 21times a week
And the value corresponds to 21/7500 = Transactions with this item / All transactions
The lift is a sign of how much our rule is strong so we're looking for rather high lifts"""

#Visualising the results:
results=list(rules)
# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])