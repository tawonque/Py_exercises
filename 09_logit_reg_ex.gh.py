# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:42:15 2016

@author: Tavo
"""


'''Assignment --- logistic regression'''
'''Marketing campaign of a Portuguese bank'''
'''INCOMPLETE BUT GETTING THERE!!!!!!'''
#Loading packages
import os
import numpy as np
import pandas as pd

from sklearn import linear_model as lm, metrics, cross_validation as cv,\
                    grid_search, feature_selection
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib qt

sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 2
})

#Data loading and cleaning (as needed)
###three files, let's open them and see their content first...

###full = pd.read_csv('datasets/bank/bank-full.csv') ---> for now, let's work with a subset
###names = pd.read_csv('datasets/bank/bank-names.txt') ---> an explanation of the dataset
bank = pd.read_csv('datasets/bank/bank.csv', ';')
predictors =  ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',\
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',\
       'previous', 'poutcome']
response = 'y'

#Initial exploratory analysis (summary statistics and visualisations)
bank.head() #something wrong, probably needs transposing? separator?
            #it was the separator, fixed
bank.describe() #valid for the numerical variables
bank.dtypes

###change data type of categorical variables
bank.job = bank.job.astype('category')
bank.marital = bank.marital.astype('category')
bank.education = bank.education.astype('category')
bank.default = bank.default.astype('category')
bank.housing = bank.housing.astype('category')
bank.loan = bank.loan.astype('category')
bank.contact = bank.contact.astype('category')
bank.month = bank.month.astype('category')
bank.poutcome = bank.poutcome.astype('category')

bank.y.replace('yes', 1, inplace=True)
bank.y.replace('no', 0, inplace=True)
bank.y = bank.y.astype('int64')


###let's check the categorical variables
f, xarr = plt.subplots(3, 3)
sns.countplot('job', data=bank, ax=xarr[0,0])
sns.countplot('marital', data=bank, ax=xarr[0,1])
sns.countplot('education', data=bank, ax=xarr[0,2])
sns.countplot('default', data=bank, ax=xarr[1,0])
sns.countplot('housing', data=bank, ax=xarr[1,1])
sns.countplot('loan', data=bank, ax=xarr[1,2])
sns.countplot('contact', data=bank, ax=xarr[2,0])
sns.countplot('month', data=bank, ax=xarr[2,1])
sns.countplot('poutcome', data=bank, ax=xarr[2,2])

###let's check the numerical variables
sns.pairplot(bank[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].dropna())

###some vairables could be transformed... but it makes sense or is it even needed?
bank.age.describe()
f, xarr2 = plt.subplots(2)
sns.distplot((np.log(bank.age)), ax=xarr2[0])
sns.distplot(bank.age, ax=xarr2[1])

###Some exploring...
bank['balancelog'] = np.log(bank.balance)
sns.boxplot(x='default', y='balancelog', data=bank)
defnum = bank.groupby('education').default.count()

###counting how many poeple of each education level have defaulted 
defyes = bank[bank.default == 'yes']
defnumyes = defyes.groupby('education').default.count()
educvar = bank.education.unique()
sns.barplot(educvar, defnumyes)

#Modelling (logistic regression)
###using statsmodel
model0 = smf.logit('y ~ 1', data=bank).fit()

model1 = smf.logit('y ~ age', data=bank).fit()
model1.summary()
model1.summary2()
np.exp(model1.params)

model2 = smf.logit('y ~ balance', data=bank).fit()
model3 = smf.logit('y ~ day', data=bank).fit()
model4 = smf.logit('y ~ duration', data=bank).fit()
model5 = smf.logit('y ~ campaign', data=bank).fit()
model6 = smf.logit('y ~ pdays', data=bank).fit()
model7 = smf.logit('y ~ previous', data=bank).fit()
model8 = smf.logit('y ~ job', data=bank).fit()
model9 = smf.logit('y ~ marital', data=bank).fit()
model10 = smf.logit('y ~ education', data=bank).fit()
model11 = smf.logit('y ~ default', data=bank).fit()
model12 = smf.logit('y ~ housing', data=bank).fit()
model13 = smf.logit('y ~ loan', data=bank).fit()
model14 = smf.logit('y ~ contact', data=bank).fit()
model15 = smf.logit('y ~ month', data=bank).fit()
model16 = smf.logit('y ~ poutcome', data=bank).fit()

###from checking the single variable models, we see that the lowest AIC is in
###models 4, 16, 15, 14
###we also notice that the 

modelall = smf.logit('y ~ age + job + marital + education + default + balance +\
 + housing + loan + contact + day + month + duration + campaign + pdays + previous +\
 poutcome', data=bank).fit()
np.exp(modelall.params)

#Cross-validation and regularisation

# Define stratified folds for cross-validation
kf = cv.StratifiedKFold(y, n_folds=10, shuffle=True)

#Compute the logistic regresion but using Scikit learn
X = bank[predictors]
y1 = bank.y.astype('float')
modelall_sk = lm.LogisticRegression(C=1e50)
modelall_sk.fit(X, y1) #####CANNOT FIGURE OUT THE PROBLEM HERE

# Compute accuracies and average AUC across folds
accs = cv.cross_val_score(modelall_sk, X, y, scoring='accuracy', cv=kf)
aucs = cv.cross_val_score(modelall_sk, X, y, scoring='roc_auc', cv=kf)

np.mean(aucs)
np.mean(accs)