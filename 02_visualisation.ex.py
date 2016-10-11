#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 2: Visualisation with matplotlib and Seaborn
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib qt

FHRS_URL = 'https://opendata.camden.gov.uk/api/views/ggah-dkrr/rows.csv?accessType=DOWNLOAD'

# Read in the Food Hygiene Rating Scheme data from `FHRS_URL` into a DataFrame
# called `fhrs`
>>> fhrs = pd.read_csv(FHRS_URL)

# Change the data type of 'Rating Date' to `datetime`
>>> pd.to_datetime(fhrs['Rating Date'])

# Filter `fhrs` to include only restaurants/cafés/canteens that are not exempt,
# and are not awaiting a new rating
BT = fhrs['Business Type Description'] == 'Restaurant/Cafe/Canteen'
NR = fhrs['New Rating Pending'] == False

'''first check if there is an 'exempt' value'''
fhrs['Rating Value'].value_counts()
EV = fhrs['Rating Value'] != 'Exempt'
fhrs_filt = fhrs[(BT) & (NR) & (EV)]

# Change the data type of 'Rating Value' to 'int'
fhrs_filt['Rating Value'] = fhrs_filt['Rating Value'].astype('int')

# Produce a bar plot of 'Rating Value'
'''convert the series to a value_counts df'''
Rating_bar = fhrs_filt['Rating Value'].value_counts()
Rating_bar.plot.bar()

# Create a new variable 'Rating Year' from 'Rating Date'
fhrs_filt['Rating Date'] = pd.to_datetime(fhrs_filt['Rating Date'])
fhrs_filt['Rating Year'] = fhrs_filt['Rating Date'].map(lambda x:x.year)

# Produce a box plot of 'Rating Value' grouped by 'Rating Year'
fhrs_filt.boxplot(column='Rating Value', by='Rating Year')

# Produce a scatter plot of 'Hygiene Score', 'Structural Score', 'Confidence In Management Score', and 'Rating Value'
sns.pairplot(fhrs_filt[scores].dropna())

# BONUS: Using Seaborn, produce a scatter plot of 'Hygiene Score' against
#        'Rating Value' including a linear regression line.
#        Add some jitter to prevent overplotting.
>>> sns.lmplot(x='Hygiene Score', y='Rating Value', data=fhrs_filt, x_jitter=1.25, y_jitter=0.25)


