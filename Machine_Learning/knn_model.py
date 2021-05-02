#!/usr/bin/env python
# coding: utf-8



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from numpy import std
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier




def twoClass_roc_auc_score(y_test, y_proba):   #Compute ROC_AUC score for 2 classes
    y_proba = y_proba[:,1]
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")
    return auc




# Importing the dataset
dataset = pd.read_excel ('IUGR.xlsx', sheet_name='Wk 22')
# 1 = IUGR, 0 = NORMAL, 2 = SGR
# 0 & 1 = 241 rows, altogether 349 rows
dataset_2 = dataset[dataset.labels != 2]
dataset_2 = dataset_2.reset_index(drop=True)
# From column age to column GA scan(wk) as features
# Clinical Information #8 - age, 9 - Ethnicsa
X = dataset_2.iloc[:, [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25]]  # with clinical information
#X = dataset_2.iloc[:, [8,9,10,11,12,13,14,16,17,18,19,20,21,22,25]]  # Nuchal Fold deleted
y = dataset_2.iloc[:, 7]


sc = StandardScaler()
X = sc.fit_transform(X)

print('Original dataset shape %s' % Counter(y))




scores = []
sensitivity = []
specificity = []
PPV = []
NPV = []
AUROCS = []
F1s = []

sm = SMOTE(random_state=42)
classifier = KNeighborsClassifier()

parameters = [{'n_neighbors': [2, 3, 4, 6, 8, 10, 12], 
               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'leaf_size': [10, 20, 30, 40, 50],
              }]
 
grid_search = RandomizedSearchCV(estimator = classifier,
                           param_distributions = parameters,
                           scoring = 'accuracy', n_jobs = -1) 




n=1
cv = KFold(n_splits=10)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_train))
    #grid_search = grid_search.fit(X_train, y_train)
    #classifier.set_params(**grid_search.best_params_)
    #best_param = grid_search.best_params_
    #print(best_param)

    # the below hyperparameters have been searched using randomsearch and will be used on all datasets
    if n == 1:
        best_param = {'n_neighbors': 2, 'leaf_size': 40, 'algorithm': 'auto'}
    elif n == 2:
        best_param = {'n_neighbors': 2, 'leaf_size': 10, 'algorithm': 'ball_tree'}
    elif n == 3:
        best_param = {'n_neighbors': 2, 'leaf_size': 20, 'algorithm': 'auto'}
    elif n == 4:
        best_param = {'n_neighbors': 3, 'leaf_size': 30, 'algorithm': 'brute'}
    elif n == 5:
        best_param = {'n_neighbors': 12, 'leaf_size': 20, 'algorithm': 'brute'}
    elif n == 6:
        best_param = {'n_neighbors': 3, 'leaf_size': 10, 'algorithm': 'brute'}
    elif n == 7:
        best_param = {'n_neighbors': 2, 'leaf_size': 30, 'algorithm': 'kd_tree'}
    elif n == 8:
        best_param = {'n_neighbors': 2, 'leaf_size': 50, 'algorithm': 'auto'}
    elif n == 9:
        best_param = {'n_neighbors': 2, 'leaf_size': 50, 'algorithm': 'auto'}
    else:
        best_param = {'n_neighbors': 4, 'leaf_size': 10, 'algorithm': 'brute'}
        
    classifier.set_params(**best_param)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    scores.append(classifier.score(X_test, y_test))
    confusion = confusion_matrix(y_test, y_predict, labels=[0,1])
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP/ float(TP+FP)
    recall = TP/ float(TP+FN)
    F1 = (2*(precision * recall))/(precision + recall)
    AUC = twoClass_roc_auc_score(y_test, y_proba)
    AUROCS.append(AUC)
    sensitivity.append(TP / float(FN + TP))
    specificity.append(TN / float(TN + FP))
    PPV.append(TP / float(TP+FP))
    NPV.append(TN/ float(TN+FN))
    F1s.append(F1)
    n +=1




print('accuracy:', np.mean(scores), np.std(scores))
sensitivity = np.array(sensitivity)
specificity = np.array(specificity)
PPVs = np.array(PPV)
NPVs = np.array(NPV)
AUROCS = np.array(AUROCS)
F1s = np.array(F1s)
print('sensitivity:', sensitivity[~np.isnan(sensitivity)].mean(), sensitivity[~np.isnan(sensitivity)].std())
print('specificity:', specificity[~np.isnan(specificity)].mean(), specificity[~np.isnan(specificity)].std())
print('PPV:', PPVs[~np.isnan(PPVs)].mean(), PPVs[~np.isnan(PPVs)].std())
print('NPV:', NPVs[~np.isnan(NPVs)].mean(), NPVs[~np.isnan(NPVs)].std())
print('F1:', F1s[~np.isnan(F1s)].mean(), F1s[~np.isnan(F1s)].std())
print('AUROC:', AUROCS[~np.isnan(AUROCS)].mean(), AUROCS[~np.isnan(AUROCS)].std())






