# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from numpy import mean
from numpy import std

def multiclass_roc_auc_score(y_pred, y_test):
    #y_test = to_categorical(y_test, num_classes=3)
    #y_pred = to_categorical(y_pred, num_classes=3)
    #lb = LabelBinarizer()
    #lb.fit(y_test)
    y_test = label_binarize(y_test, classes=[0,1,2])
    y_pred = label_binarize(y_pred, classes=[0,1,2])
    try:
        auc = roc_auc_score(y_test, y_pred, average="macro")
    except ValueError:
        auc = float("nan")
    
    return auc

def twoClass_roc_auc_score(y_test, y_proba):
    y_proba = y_proba[:,1]
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")
    return auc

def evaluate_AUC(y_predict, y_test):
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc
    from numpy import interp
    testy_enc = to_categorical(y_test, num_classes=3)
    y_pred_enc = to_categorical(y_predict, num_classes=3)
    n_classes=3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testy_enc[:, i], y_pred_enc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testy_enc.ravel(), y_pred_enc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc["macro"]


# Importing the dataset

dataset = pd.read_excel ('IUGR.xlsx', sheet_name='Wk 22')
dataset_2 = dataset[dataset.labels != 2]
dataset_2 = dataset_2.reset_index(drop=True)

X = dataset_2.iloc[:, [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25]]  # All parameters
#X = dataset_2.iloc[:, [8,9,10,11,12,13,14,16,17,18,19,20,21,22,25]]  # Nuchal Fold deleted
y = dataset_2.iloc[:, 7]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV
parameters = [{'kernel': ['linear'], 'C': [1,10,100,1000,10000,100000]}, {'kernel':['rbf'], 'C': [1,10,100,1000,10000,100000],'gamma':[0.1, 0.01, 0.001, 0.0001, 0.00001]}]
 
grid_search = RandomizedSearchCV(estimator = SVC(),
                           param_distributions = parameters,
                           scoring = 'accuracy', n_jobs = -1) 

# Fitting SVM to the Training set using CV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
scores = []
sensitivity = []
specificity = []
PPV = []
NPV = []
AUROCS = []
F1s = []
classifier = SVC()
n=1
cv = KFold(n_splits=10)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_train))
    grid_search = grid_search.fit(X_train, y_train)
    classifier.set_params(**grid_search.best_params_)
    print(grid_search.best_params_)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    scores.append(classifier.score(X_test, y_test))
    confusion = confusion_matrix(y_test, y_predict, labels=[0,1])
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP/ float(TP+FP)
    recall = TP/ float(TP+FN)
    F1 = (2*(precision * recall))/(precision + recall)
    sensitivity.append(TP / float(FN + TP))
    specificity.append(TN / float(TN + FP))
    PPV.append(TP / float(TP+FP))
    NPV.append(TN/ float(TN+FN))
    F1s.append(F1)
    n +=1

print('accuracy:', np.mean(scores), np.std(scores))
sensitivity = np.array(sensitivity)
specificity = np.array(specificity)
PPV = np.array(PPV)
NPV = np.array(NPV)
F1s = np.array(F1s)
print('sensitivity:', sensitivity[~np.isnan(sensitivity)].mean(), sensitivity[~np.isnan(sensitivity)].std())
print('specificity:', specificity[~np.isnan(specificity)].mean(), specificity[~np.isnan(specificity)].std())
print('PPV:', PPV[~np.isnan(PPV)].mean(), PPV[~np.isnan(PPV)].std())
print('NPV:', NPV[~np.isnan(NPV)].mean(), NPV[~np.isnan(NPV)].std())
print('F1:', F1s[~np.isnan(F1s)].mean(), F1s[~np.isnan(F1s)].std())


