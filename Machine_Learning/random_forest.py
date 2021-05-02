# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:15 2020

@author: Leonard Teng
"""

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import argmax
from sklearn.metrics import roc_auc_score

def twoClass_roc_auc_score(y_test, y_proba):
    y_proba = y_proba[:,1]
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")
    return auc

# Importing the dataset
dataset = pd.read_excel ('IUGR.xlsx', sheet_name='Wk 22')
dataset_2 = dataset[dataset.labels != 2]
dataset_2 = dataset_2.reset_index(drop=True)
#X = dataset_2.iloc[:, [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25]]  # All parameters
X = dataset_2.iloc[:, [8,9,10,11,12,13,14,16,17,18,19,20,21,22,25]]  # with Nuchal Fold deleted
y = dataset_2.iloc[:, 7]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import RandomizedSearchCV

print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    criterion='entropy',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
)

param_grid = {'n_estimators': [10,20,30], 'max_depth': [4,5,6,7]}

grid_search = RandomizedSearchCV(estimator = clf,
                           param_distributions = param_grid,
                           scoring = 'accuracy') 

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
scores = []
sensitivity = []
specificity = []
PPV = []
NPV = []
AUROCS = []
F1s = []

n = 1
cv = KFold(n_splits=10, random_state=0)
for train_index, test_index in cv.split(X):
    print('============================================')
    print('FOLD ', n)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_train))
    grid_search = grid_search.fit(X_train, y_train)
    clf.set_params(**grid_search.best_params_)
    print(grid_search.best_params_)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    scores.append(clf.score(X_test, y_test))
    #confusion = confusion_matrix(y_test, y_predict)
    confusion = confusion_matrix(y_test, y_predict, labels=[0,1])
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    precision = TP/ float(TP+FP)
    recall = TP/ float(TP+FN)
    F1 = (2*(precision * recall))/(precision + recall)
    AUC = twoClass_roc_auc_score(y_test, y_proba)
    sensitivity.append(TP / float(FN + TP))
    specificity.append(TN / float(TN + FP))
    PPV.append(TP / float(TP+FP))
    NPV.append(TN/ float(TN+FN))
    AUROCS.append(AUC)
    F1s.append(F1)
    n += 1
    

print('accuracy:', np.mean(scores), np.std(scores))
sensitivity = np.array(sensitivity)
specificity = np.array(specificity)
PPV = np.array(PPV)
NPV = np.array(NPV)
AUROCS = np.array(AUROCS)
F1s = np.array(F1s)
print('sensitivity:', sensitivity[~np.isnan(sensitivity)].mean(), sensitivity[~np.isnan(sensitivity)].std())
print('specificity:', specificity[~np.isnan(specificity)].mean(), specificity[~np.isnan(specificity)].std())
print('PPV:', PPV[~np.isnan(PPV)].mean(), PPV[~np.isnan(PPV)].std())
print('NPV:', NPV[~np.isnan(NPV)].mean(), NPV[~np.isnan(NPV)].std())
print('AUROC:', AUROCS[~np.isnan(AUROCS)].mean(), AUROCS[~np.isnan(AUROCS)].std())
print('F1:', F1s[~np.isnan(F1s)].mean(), F1s[~np.isnan(F1s)].std())
