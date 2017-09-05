#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:35:37 2017

@author: tingzhu
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve, classification_report
from sklearn.model_selection import GridSearchCV

def classify_randomForest(data):
    
    # split the raw data into training and test datasets 
    # for cross validation: train/test = 80/20 split (train/test)
    
    
    msk = np.random.rand(len(data)) < 0.8 # choose 80% of data to be training data
    train_set=data[msk]
    test_set=data[~msk]
    
    X_train=train_set.iloc[:,3:]
    Y_train=train_set.iloc[:,2]
    
    X_test=test_set.iloc[:,3:]
    Y_test=test_set.iloc[:,2]
    #print('Training data:')    
    #print(Y_train.value_counts())
    #print('Test data:')
    #print(Y_test.value_counts())

    clf = RandomForestClassifier(n_estimators=20)
    param_grid = {"max_depth": [None],
                  "max_features": [10,25],
                  "min_samples_split": [15,25],
                  "min_samples_leaf": [5,20],
                  "bootstrap": [True],
                  "criterion": ["entropy"]}
    
    # run grid search for hyperparameter search
    # K-fold is 5 for additional cross validation
    grid_search = GridSearchCV(clf, param_grid=param_grid,cv=5)
    
    grid_search.fit(X_train, Y_train)
    
    print("\n------ Random Forest Model Result ------")
    print()
    print('Search grid of parameters')
    print(param_grid)
    print()
    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = grid_search.cv_results_['mean_test_score']
#    stds = grid_search.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))

    Y_true, Y_pred = Y_test, grid_search.predict(X_test)
    print(classification_report(Y_true, Y_pred))
    
   # Y_pred_prob=grid_search.predict_proba(X_test)
    
    fpr, tpr, _ = roc_curve(Y_true, Y_pred)
    auc = roc_auc_score(Y_true, Y_pred)
    plt.plot(fpr, tpr, label='test_set (AUC=%.3f)'%(auc))
    plt.legend(fontsize=12)
    plt.title('ROC curve for RF classifier')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig('./ROC_curve.png',dpi=600)
    #plt.show()
    
    
    return grid_search