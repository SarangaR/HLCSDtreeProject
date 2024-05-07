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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

import joblib
import os

data = pd.read_parquet("RT_IOT2022.parquet")

X_train, X_test, y_train, y_test = pp.preprocess(data)

## Gradient Boosting
grad = GradientBoostingClassifier(random_state=42)

grad.fit(X_train, y_train)

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, grad.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, grad.predict(X_train))}")

joblib.dump(grad, "gradient_boosting.pkl")

## XG Boosting
xg = XGBClassifier(random_state=42)
xg.fit(X_train, y_train)

print(f"Validation Acc: {sklearn.metrics.accuracy_score(y_test, xg.predict(X_test))}")
print(f"Training Acc: {sklearn.metrics.accuracy_score(y_train, xg.predict(X_train))}")