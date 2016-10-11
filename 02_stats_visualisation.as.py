#!/usr/bin/env python

'''
GA Data Science Q2 2016

Assignment 2: Summary statistics and visualisation
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

%matplotlib qt

'''
Exercise 1
'''

# Read in the General Household Survey 2005 data
ghs = pd.read_csv('datasets/ghs2005.csv')

# Change the data type of 'sex' to `int`
ghs['sex'] = ghs.sex.astype('int')
ghs['sex'].dtype

# Recode missing values in the variables 'drating' and 'workhrs'
'''this would change to 99 the NaN but I left them as Nan'''
'''ghs.drating.replace(np.NaN, 99, inplace=True)
ghs.workhrs.replace(np.NaN, 99, inplace=True)'''

# Compute the mode of 'drating'
ghs_nn = ghs['drating'].notnull()
ghs[ghs_nn].drating.mode()

# Compute the mean and standard deviation of 'drating'
ghs[ghs_nn].drating.mean()
ghs[ghs_nn].drating.std()

# Compute the median of 'drating'
ghs[ghs_nn].drating.median()

# Repeat excluding zeros
'''and also excluding null values?'''
ghs_nnnz = ghs['drating'] != 0
ghs[ghs_nnnz].drating.dropna()

'''
Exercise 2
'''

# Visualise the distribution of 'drating' (histogram and density estimate)
ghs['drating'].plot.hist()

# Repeat excluding zeros
ghs_nz = ghs[ghs_nnnz]
ghs_nz.drating.plot.hist()

# Repeat excluding zeros and applying a logarithmic transformation
>>> ghs_nz['log drating'] = np.log(ghs_nz['drating'])
>>> ghs_nz['log drating'].plot.hist(20)


# Produce a box plot of 'drating' grouped by 'sex'
ghs_nz.boxplot(column=['drating'], by='sex')


# BONUS: Repeat after applying a logarithmic transformation to 'drating'
>>> ghs_nz.boxplot(column=['log drating'], by='sex')

# Produce a scatter plot of 'drating', 'age', and 'workhrs'
ghs_nz.plot.scatter(x='drating' , y='age', c='workhrs')

# Compute the correlation matrix of 'drating', 'age', and 'workhrs'


# BONUS: Represent the correlation matrix using a heat map

# BONUS: Formally test the hypothesis that 'drating' differs by 'sex' using a
#        two-sample t-test.

# BONUS: Repeat after applying a logarithmic transformation.

# BONUS: Repeat using a Wilcoxon rank-sum test, and observe the effect of the
#        logarithmic transformation in this case.

