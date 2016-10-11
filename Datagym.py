# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:32:58 2016

@author: Tavo
"""

''''Datagym'''

import os
import pandas as pd
import numpy as np

from sklearn import linear_model as lm, metrics, cross_validation as cv,\
                    grid_search, feature_selection

import matplotlib.pyplot as plt
import seaborn as sns


custom = pd.read_csv(os.path.join('datasets', 'datagym', 'datagym_customers.csv'))
visit = pd.read_csv(os.path.join('datasets', 'datagym', 'datagym_visits.csv'))

custom = custom.dropna()
visit = visit.dropna()

visit.date = pd.to_datetime(visit.date)
visit.sort_values(['customer_id', 'date'], inplace=True)

visit = visit.set_index(visit.date)
visit['interval'] = visit.date - visit.date.shift(1)
visit.dropna(inplace=True)

visit['interhours'] = visit.interval / np.timedelta64(1, 'h')
visit = visit[visit.interhours > 0]
sns.boxplot(x='xxxxgroups of ten clients...xxxxx', y='interhours', data=visit)


visitgap = visit.groupby('customer_id').interhours.mean()
visitgap = visitgap.sort_values(axis=0)

visitgap = pd.DataFrame(visitgap)
visitgap['interhours_log'] = np.log(visitgap.interhours)



---------o---------o---------o---------
visit.groupby('customer_id').interval.mean()
hires_diff = hires_diff - hires_diff.shift(1)
hires_diff.dropna(inplace=True)
In [74]: td / np.timedelta64(1, 's')
bsa2014[bsa2014.RAge < 20]             # (also in a single step)

sns.barplot(x='Countrygp', y='LIFEXP', data=wbwdi)
