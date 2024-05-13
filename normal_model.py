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

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import joblib
import os
import json

data = pd.read_parquet("RT_IOT2022.parquet")

X_train, X_test, y_train, y_test = pp.preprocess(data)

print("=========== Decision Tree ===========")

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, clf.predict(X_train))}")

joblib.dump(clf, "models/decision_tree.pkl")

with open('accuracies.json', 'r+') as f:
    data = json.load(f)
    data['decision_tree']['train'] = sklearn.metrics.accuracy_score(y_train, clf.predict(X_train))
    data['decision_tree']['test'] = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

## hyperparameter tuning
print("\n=========== Decision Tree Tuning ===========")

params_grid = { 'criterion': ['gini', 'entropy'], 'max_depth': randint(4, 40), 'max_leaf_nodes': randint(1000, 20000), 'min_samples_leaf': randint(2, 40), 'min_samples_split': randint(1, 40) }

search = HalvingRandomSearchCV(clf, params_grid, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)
clf_tuned = search.best_estimator_

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, clf_tuned.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, clf_tuned.predict(X_train))}")

joblib.dump(clf_tuned, "models/decision_tree_tuned.pkl")

with open('accuracies.json', 'r+') as f:
    data = json.load(f)
    data['decision_tree_tuned']['train'] = sklearn.metrics.accuracy_score(y_train, clf_tuned.predict(X_train))
    data['decision_tree_tuned']['test'] = sklearn.metrics.accuracy_score(y_test, clf_tuned.predict(X_test))
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()
