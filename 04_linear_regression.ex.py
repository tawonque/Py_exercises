#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 4: Linear regression using StatsModels
'''

import os

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.gofplots as smg

from pandas.tools.plotting import autocorrelation_plot

%matplotlib qt

sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 14
})

# Read in the EU referendum data from the 'Web scraping' code walk-through
brexit = pd.read_csv('datasets/brexit.csv')

# Change the data type of 'date' to `datetime`
brexit['date'] = pd.to_datetime(brexit['date'])

# Create a new variable 't' representing the years between 2015-01-01 and 'date'
# (Hint: convert to `timedelta64[D]` first, then divide by 365.25)
'''start date setting'''
brexit['start'] = '2015-01-01'
brexit['start'] = pd.to_datetime(brexit['start'])
brexit['start'] = pd.to_timedelta(brexit['start'])

'''calculate interval and then convert into years'''
brexit['datedelta'] = pd.to_timedelta(brexit['date'])
brexit['days_interval'] = brexit['datedelta']-brexit['start']
brexit['interval'] = brexit['days_interval'] / pd.Timedelta(days=365.25)

t = brexit['interval']

sns.lmplot(x='t', y='stay', data=brexit)

# Build a regression model for 'stay' versus 't'
model1 = smf.ols('stay ~ t', data=brexit).fit()

# Examine the model output
model1.summary()
model1.summary2()

# Produce the following diagnostic plots:

# * Predicted versus observed
sns.jointplot(brexit['stay'], model1.fittedvalues)

# * Residuals versus predicted
sns.jointplot(brexit['stay'], model1.resid)

# * Residuals versus 't'
sns.jointplot(t, model1.resid)

# * Autocorrelation plot
autocorrelation_plot(model1.resid)

# * Normal Q-Q plot for (Studentised) residuals
st_resid = model1.get_influence().get_resid_studentized_external()
qq = smg.qqplot(st_resid)
smg.qqline(qq.gca(), '45')

# BONUS: Build a second regression model for 'stay' versus 't' and 'pollster',
#        and re-run all of the above
model2 = smf.ols('stay ~ t + pollster', data=brexit).fit()

st_resid2 = model2.get_influence().get_resid_studentized_external()
qq2 = smg.qqplot(st_resid2)
smg.qqline(qq2.gca(), '45')

