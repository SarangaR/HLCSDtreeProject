import sklearn
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import randint

import sklearn.metrics
import preprocessing as pp

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import joblib
import os

data = pd.read_parquet("RT_IOT2022.parquet")

X_train, X_test, y_train, y_test = pp.preprocess(data)

clf = HistGradientBoostingClassifier(verbose=5, random_state=42)

# last_accuracy = 0
# accuracy = 0

# if os.path.exists("model.pkl"):
#     clf = joblib.load("model.pkl")
#     clf.fit(X_train, y_train)
#     accuracy = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))
#     last_accuracy = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))
# # else:
# #     clf.fit(X_train, y_train)
# #     joblib.dump(clf, "model.pkl")

# while accuracy < last_accuracy:
#     clf.fit()
#     # accuracy = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))
#     # print(accuracy)
#     # clf = joblib.load("model.pkl")
#     # clf.fit(X_train, y_train)
#     # last_accuracy = accuracy
#     # joblib.dump(clf, "model.pkl")

params_dist = { 'random_state': randint(low=1, high=100) }

clf_tuned = clf

print(params_dist)

random_search = RandomizedSearchCV(clf_tuned, params_dist, cv=7, verbose=5, n_jobs=3)

random_search.fit(X_train, y_train)

print(random_search.best_estimator_)

best_tuned_clf = random_search.best_estimator_ 
print(sklearn.metrics.accuracy_score(y_test, best_tuned_clf.predict(X_test)))

print(sklearn.metrics.classification_report(y_test, clf.predict(X_test)))
print(sklearn.metrics.classification_report(y_test, best_tuned_clf.predict(X_test)))

# print(data.index)
# export_graphviz(clf, out_file='forest.dot', feature_names=list(X_train), class_names=data.columns.values.tolist(), rounded=True, filled=True )