#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:21:38 2018

@author: hamzatazi
"""

import pandas as pd

main_file_path = '../input/train.csv'
train = pd.read_csv(main_file_path)
y_train=train.SalePrice
melbourne_predictors=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X_train=train[melbourne_predictors]

test = pd.read_csv('../input/test.csv')
X_test = test[melbourne_predictors]

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(X_train,y_train)
predicted_prices=regressor.predict(X_test)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)