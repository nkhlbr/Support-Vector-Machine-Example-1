# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:38:12 2021

@author: nikhil.barua
"""


#Understanding the data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.keys()


print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

df_feat.info()

df_feat.describe()

df_feat.head()

cancer['target']

cancer['target_names']

#Training and Testing the data

from sklearn.model_selection import train_test_split, GridSearchCV

X = df_feat
y = cancer['target']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

#Evaluate the predictions

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test,predictions))


from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000], 'gamma': [1,.1,.01,.001,.0001]}

 
grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))




