# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:15:58 2016

@author: Tavo
"""


'''Bike project data extracting and editing'''

'''Packages'''
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Panel
from datetime import datetime

from sklearn import preprocessing, metrics

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, svm, cross_validation as cv, grid_search


from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure, show, rc
import seaborn as sns

bike = pd.read_csv('bike_project/train.csv')
bike.datetime = pd.to_datetime(bike.datetime)
bike.set_index('datetime', inplace=True)
bike['weekday'] = bike.index.weekday.astype('int')

#plt.plot_date(y=bike['count'], x=bike.index, data=bike)
#sns.boxplot(x='date', y='count', data=bike)

Dsum = bike.resample('D').sum()
Dmean = bike.resample('D').mean()

#plt.plot_date(y=Dsum['count'], x=Dsum.index, data=bike)
#plt.plot_date(y=Dmean['count'], x=Dsum.index, data=bike)

bikedayDsum = Dsum[['casual', 'registered', 'count']].dropna()
bikedayDmean = Dmean[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',\
 'humidity', 'windspeed', 'weekday']].dropna()


###====== Input ======
X = bikedayDmean
y = bikedayDsum[['count']]
X = pd.concat([X, y], axis=1)
X = X.as_matrix()

###====== Scale X ======
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#until here...

###======= NN Model // not ideal for a continuous prediction =======
## Initialise neural network
nn = Sequential() #Just a container for the neural network

## Input layer feeding into hidden layer with 5 neurons (sigmoid activation)
nn.add(Dense(input_dim=X.shape[1], output_dim=5, activation='linear'))
nn.add(Dense(output_dim=4, activation='linear'))
nn.add(Dense(output_dim=1, activation='linear'))


## Compile
nn.compile(loss='mean_squared_error', optimizer='rmsprop')

## Inspect weights before training
nn.get_weights()

## Train the networks
nn.fit(X_scaled, y, batch_size=50, nb_epoch=50, validation_split=0.2)

## Inspect weights after training
nn.get_weights()

## Use network to predict probabilities
pred_train = nn.predict_proba(X_scaled)[:,0]
#problem is, it is difficult to interpret this output. 

## Compute AUC
#metrics.roc_auc_score(y, pred_probs)


'''
###======= Time series exercise ======

X = bikedayDmean
y = bikedayDsum[['count']]


# Extract first column (daily hires) and convert to 'float'
y = y.astype('float')

# Apply logarithmic transformation
y = np.log10(y)


ssvc = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('svr', svm.SVR())
])

# Fit the model
ssvc.fit(X, y)

# Compute MSE for split
split=5
cv.cross_val_score(ssvc, X, y, scoring='mean_squared_error', cv=split)

# Determine ‘optimal’ kernel and value of C by cross-validation
gs = grid_search.GridSearchCV(
    estimator=ssvc,
    param_grid={
        'svr__C': [1e-15, 0.0001, 0.001, 0.01, 0.1, 1, 10],
        'svr__kernel': ['linear', 'rbf']},
    scoring='mean_squared_error',
    cv=split)
gs.fit(X, y)

gs.best_score_
gs.best_estimator_
gs.grid_scores_

# Plot original time series and prediction
y['Prediction'] = gs.best_estimator_.predict(X)
y.plot()
'''
'''Now I need to load the train data and perform the prediction on the same model. 
Once ready, plot both on the same graph and compre to the real data.'''

### ====== Linear regression ======

bikex = pd.concat([bikedayDmean, bikedayDsum])
bikex = bikex.dropna()





