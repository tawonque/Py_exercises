#!/usr/bin/env python

'''
=== GA Data Science Q2 2016 ===

Assignment 1: Introduction to pandas
'''

import os

import numpy as np
import pandas as pd

'''
Exercise 1
'''

# Read in the World Bank World Development Indicators data from `wbwdi.csv`
# into a DataFrame called `wbwdi`
>>> wbwdi = pd.read_csv('datasets/wbwdi.csv', index_col=0)

# Print the ‘head’ and ‘tail’
>>> wbwdi.head()
>>> wbwdi.tail()

# Examine the row names (index), data types, and shape
>>> wbwdi.index
>>> wbwdi.dtypes
>>> wbwdi.shape

# Print the 'LIFEXP' Series
>>> wbwdi['LIFEXP']

# Calculate the mean 'LIFEXP' for the entire dataset
>>> wbwdi['LIFEXP'].mean()

# Count the number of occurrences of each 'Countrygp'
>>> wbwdi.index.value_counts()

# BONUS: Display only the number of rows of `wbwdi`
>>> len(wbwdi) '''not sure about the question nor about the answe!'''

# BONUS: Display the 3 most frequent values of 'Countrygp'
>>> wbwdi_Countrygp_counts = wbwdi.index.value_counts()
>>> wbwdi_Countrygp_counts.sort_values(ascending=False).head(3)

'''
Exercise 2
'''

# Filter `wbwdi` to only include African countries
>>> wbwdi[wbwdi.index == 2.0]

# Filter `wbwdi` to only include African countries with LIFEXP > 60
>>> wbwdi[(wbwdi.index == 2.0) & (wbwdi.LIFEXP >60)]

# Calculate the mean 'LIFEXP' for all of Africa
>>> wbwdi[wbwdi.index == 2.0].LIFEXP.mean()

# Determine which 10 countries have the highest LIFEXP
>>> wbwdi.sort_values('LIFEXP', ascending=False).head(10)

# BONUS: Sort `wbwdi` by 'Countrygp' and then by 'LIFEXP' (in a single command)
>>> wbwdi.sort_index().sort_values('LIFEXP')
'''or should it be groupby and then sort???...'''


# BONUS: Filter `wbwdi` to only include African or Middle Eastern countries
#        without using `|`.

'''
Exercise 3
'''

# Count the number of missing values in each column of `wbwdi`
>>> wbwdi.isnull().sum()

# Show only countries for which 'LIFEXP' is missing
>>> wbwdi[wbwdi.LIFEXP.isnull()]

# How many rows remain if you drop all rows with any missing values?
>>> wbwdi.dropna().shape

# BONUS: Create a new column called 'initial' that contains the first letter of
#        the country name (e.g., 'A' for Afghanistan)
>>> wbwdi['initial'] = wbwdi.Country.str[0:1]

'''
Exercise 4
'''

# Calculate the mean 'LIFEXP' by 'Countrygp'
>>> wbwdi.pivot_table(values='LIFEXP', index=[wbwdi.index])

# Calculate the minimum and maximum 'LIFEXP' by 'Countrygp'
>>> wbwdi.pivot_table(values='LIFEXP', index=[wbwdi.index], aggfunc='min') 

# BONUS: Cross-tabulate 'Countrygp' and 'initial'

# BONUS: Calculate the median 'LIFEXP' for each combination of 'Countrygp' and
#        'initial'
>>>wbwdi.pivot_table(values='LIFEXP', index=[wbwdi.index, 'initial'], aggfunc='median')


