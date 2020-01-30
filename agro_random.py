# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:00:04 2020

@author: naif
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Book3.csv')
X=dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11]].values
y=dataset.iloc[:,[12]].values
#print(X)
#print(y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0]= le.fit_transform(X[:,0])
print(X[:,0])
X[:,1]= le.fit_transform(X[:,1])
print(X[:,1])
X[:,7]= le.fit_transform(X[:,7])
print(X[:,7])
X[:,8]= le.fit_transform(X[:,8])
print(X[:,8])
#print(X)
#print(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#print(X_train)
#print(y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import f1_score
f1_score=f1_score(y_test, y_pred)
print(f1_score)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)
#print(accuracy)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
