import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, HalvingRandomSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

import joblib
import os
import json

data = pd.read_parquet("RT_IOT2022.parquet")

X_train, X_test, y_train, y_test = pp.preprocess(data)

## Gradient Boosting
print("=========== Gradient Boosting ===========")
grad = GradientBoostingClassifier(random_state=42, verbose=2)

grad.fit(X_train, y_train)

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, grad.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, grad.predict(X_train))}")

joblib.dump(grad, "models/gradient_boosting.pkl")

with open('accuracies.json', 'r+') as f:
    data = json.load(f)
    data['gradient_boosting']['train'] = sklearn.metrics.accuracy_score(y_train, grad.predict(X_train))
    data['gradient_boosting']['test'] = sklearn.metrics.accuracy_score(y_test, grad.predict(X_test))
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

# Hyperparameter tuning with HalvingRandomSearchCV
grad = joblib.load("models/gradient_boosting.pkl")
print("\n=========== Gradient Boosting Tuning ===========")
params_grid = { 'n_estimators': randint(500, 1000, 2), 'learning_rate': [0.01, 0.1, 0.5, 1], 'max_depth': randint(2, 10), 'min_samples_split': randint(2, 10), 'min_samples_leaf': randint(1, 10) }

search = RandomizedSearchCV(grad, params_grid, n_jobs=-1, random_state=42, verbose=2, cv=7)
search.fit(X_train, y_train)
grad_tuned = search.best_estimator_

joblib.dump(grad_tuned, "models/gradient_boostin_tuned.pkl")

with open('accuracies.json', 'r+') as f:
    data = json.load(f)
    data['gradient_boosting_tuned']['train'] = sklearn.metrics.accuracy_score(y_train, grad.predict(X_train))
    data['gradient_boosting_tuned']['test'] = sklearn.metrics.accuracy_score(y_test, grad.predict(X_test))
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, grad_tuned.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, grad_tuned.predict(X_train))}")

## Ada Boosting
print("\n=========== Ada Boosting ===========")
clf_tuned = joblib.load("models/decision_tree_tuned.pkl")
ada = AdaBoostClassifier(clf_tuned, random_state=42, algorithm='SAMME')

ada.fit(X_train, y_train)

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, ada.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, ada.predict(X_train))}")

joblib.dump(ada, "models/ada_boosting.pkl")

with open('accuracies.json', 'r+') as f:
    data = json.load(f)
    data['ada_boosting']['train'] = sklearn.metrics.accuracy_score(y_train, ada.predict(X_train))
    data['ada_boosting']['test'] = sklearn.metrics.accuracy_score(y_test, ada.predict(X_test))
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

# Hyperparameter tuning with HalvingRandomSearchCV
print("\n=========== Ada Boost Tuning ===========")
params_grid = { 'n_estimators': randint(500, 1000, 2), 'learning_rate': [0.01, 0.1, 0.5, 1]}

search = RandomizedSearchCV(ada, params_grid, random_state=42, verbose=3, cv=7, n_jobs=2)
search.fit(X_train, y_train)
ada_tuned = search.best_estimator_

joblib.dump(ada_tuned, "models/ada_boosting_tuned.pkl")

with open('accuracies.json', 'r+') as f:
    data = json.load(f)
    data['ada_boosting_tuned']['train'] = sklearn.metrics.accuracy_score(y_train, ada_tuned.predict(X_train))
    data['ada_boosting_tuned']['test'] = sklearn.metrics.accuracy_score(y_test, ada_tuned.predict(X_test))
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, ada_tuned.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, ada_tuned.predict(X_train))}")