# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:22:53 2020

@author: Leonard Teng
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import initializers
from numpy import mean
from numpy import std
import numpy as np
#from numpy import array
from numpy import argmax
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

#Resample Dataset
def resample(X, y):
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    print('Original dataset shape %s' % Counter(y))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res, y_res

#calculate sensititiviy, specificity, NPV and PPV
def evaluate_cm(y_pred, testy):
    y_pred_convert = argmax(y_pred, axis=1)
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(testy, y_pred_convert)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    PPV = TP/ float(TP+FP)
    NPV = TN/ float(TN+FN)
    precision = TP/ float(TP+FP)
    recall = TP/ float(TP+FN)
    F1 = (2*(precision * recall))/(precision + recall)
    return sensitivity, specificity, NPV, PPV, F1

def twoClass_roc_auc_score(y_test, y_proba):
    y_proba = y_proba[:,1]
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")
    return auc

# evaluate a single mlp model
def evaluate_model(trainX, trainy, testX, testy, input_dim, batch_size, epochs):
    # encode targets
    trainy_enc = to_categorical(trainy)
    testy_enc = to_categorical(testy)
    # define model
    model = Sequential()
    model.add(Dense(24, input_dim=input_dim+2, activation='relu', init=initializers.Ones()))
    model.add(Dense(24, activation='relu', init=initializers.Ones()))
    model.add(Dense(2, activation='softmax', init=initializers.Ones()))  #SN
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy_enc, epochs=epochs, batch_size=batch_size, verbose=0)
    # evaluate the model
    _, test_acc = model.evaluate(testX, testy_enc, verbose=0)
    y_pred = model.predict(testX)
    return model, test_acc, y_pred

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# generate 2d classification dataset
dataset = pd.read_excel('IUGR.xlsx', sheet_name='Wk 22')
dataset_2 = dataset[dataset.labels != 2]
dataset_2 = dataset_2.reset_index(drop=True)
#X = dataset_2.iloc[:, [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25]]  # All parameters
X = dataset_2.iloc[:, [8,9,10,11,12,13,14,16,17,18,19,20,21,22,25]]  # Nuchal Fold deleted
y = dataset_2.iloc[:, 7]
X = X.to_numpy()   #convert to array
y = y.to_numpy()
input_dim = int(X.size/X.shape[0])

 #Encoding Race (Only for with clinical information)
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]


def create_model():
    # create model
    model=Sequential()
    model.add(Dense(24, input_dim=17, activation='relu', init=initializers.Ones()))
    model.add(Dense(24, activation='relu', init=initializers.Ones()))
    model.add(Dense(2, activation='softmax', init=initializers.Ones()))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

param_dist = dict(
            batch_size=(np.arange(10,100).tolist()),
            epochs=(np.arange(10, 800).tolist())
        )
model = KerasClassifier(build_fn=create_model, verbose=0)
#grid = GridSearchCV(estimator=model, param_grid=param_dist, scoring = 'accuracy', verbose=2, n_jobs=-1) # if use gridsearch    
grid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, scoring = 'accuracy', verbose=2, n_jobs=-1)

# prepare the k-fold cross-validation configuration
n_folds = 10
n = 1
kfold = KFold(n_folds, True, 1)
# cross validation estimation of performance
output_text = ''
scores, members, AUROCS, sensitivityS, specificityS, NPVs, PPVs, F1s, epoches, batch_sizes = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()

for train_ix, test_ix in kfold.split(X):
        
    print('============================================')
    print('FOLD ', n)
    # select samples
    trainX, trainy = X[train_ix], y[train_ix]
    trainX, trainy = resample(trainX, trainy)
    testX, testy = X[test_ix], y[test_ix]


    grid_result = grid.fit(trainX, trainy)
    print(grid_result.best_params_)
    batch_size = grid_result.best_params_['batch_size']
    epochs = grid_result.best_params_['epochs']

    
    # evaluate model
    model, test_acc, y_pred = evaluate_model(trainX, trainy, testX, testy, input_dim, batch_size, epochs)
    AUC = twoClass_roc_auc_score(testy, y_pred)
    sensitivity, specificity, NPV, PPV, F1 = evaluate_cm(y_pred, testy)
    print('>%.3f' % test_acc)
    
    msg = 'CV ' + str(n) + ', best_batch_size ' + str(batch_size) + ', epochs, ' + str(epochs) + ', test_acc, ' + str(round(test_acc, 3)) + '\n'
    
    n += 1

    output_text = output_text + msg
    epoches.append(epochs)
    batch_sizes.append(batch_size)
    scores.append(test_acc)
    members.append(model)
    sensitivityS.append(sensitivity)
    specificityS.append(specificity)
    NPVs.append(NPV)
    PPVs.append(PPV)
    F1s.append(F1)

    output_file = open('output.txt', 'w')
    output_file.write(output_text)
    output_file.close()
    
# summarize expected performance

F1s = np.array(F1s)
print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
print('Estimated sensitivity %.3f (%.3f)' % (mean(sensitivityS), std(sensitivityS)))
print('Estimated specificity %.3f (%.3f)' % (mean(specificityS), std(specificityS)))
print('Estimated PPV %.3f (%.3f)' % (mean(PPVs), std(PPVs)))
print('Estimated NPV %.3f (%.3f)' % (mean(NPVs), std(NPVs)))
print('Estimated AUROC %.3f (%.3f)' % (mean(AUROCS), std(AUROCS)))
print('Estimated F1 %.3f (%.3f)' % (F1s[~np.isnan(F1s)].mean(), F1s[~np.isnan(F1s)].std()))
os.system('spd-say -i 100 "Your program is Finish"')

