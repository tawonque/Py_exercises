#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 5: Logistic regression using StatsModels
'''

import os

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib qt

BANKNOTES_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'

var_names = ['wavelet_var', 'wavelet_skew', 'wavelet_kurt', 'entropy', 'forged']
predictors = 'wavelet_var', 'wavelet_skew', 'wavelet_kurt', 'entropy'


# Read in the Banknote Authentication dataset
bn = pd.read_csv(BANKNOTES_URL)
bn.columns = var_names


# Explore data visually
sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 14
})

sns.boxplot(bn[[predictors]].dropna())
sns.boxplot(x='forged', y='wavelet_var', data=bn)
sns.boxplot(x='forged', y='wavelet_skew', data=bn)
sns.boxplot(x='forged', y='wavelet_kurt', data=bn)
sns.boxplot(x='forged', y='entropy', data=bn)

sns.distplot(bn.wavelet_skew.dropna(), kde=False)
sns.pairplot(bn[[predictors]].dropna())

# Build a logistic regression model without predictors
model1 = smf.logit('forged ~ 1', data=bn).fit()

# Build one logistic regression model for each predictor
modela = smf.logit('forged ~ wavelet_var', data=bn).fit()
modelb = smf.logit('forged ~ wavelet_skew', data=bn).fit()
modelc = smf.logit('forged ~ wavelet_kurt', data=bn).fit()
modeld = smf.logit('forged ~ entropy', data=bn).fit()
model2 = smf.logit('forged ~ wavelet_var + wavelet_skew + wavelet_kurt + \
entropy', data=bn).fit()



# Select the ‘best’ predictor based on the AIC, and build three models including
# this variable and each of the remaining three
# Rule of thumb: ΔAIC < 2  = No difference, prefer model with less predictors
#                     < 6  = Model with lower AIC is preferred (assuming large N)
#                     < 10 = Model with lower AIC is preferred (assuming small N)
#                     ≥ 10 = Model with lower AIC is strongly preferred (always)

# Repeat, building two models including the two ‘most predictive’ variables and
# each of the remaining predictors

# Finally, build the last model including all predictors

# Print out and interpret the coefficients of the ‘most predictive’ model

