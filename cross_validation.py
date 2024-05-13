import joblib
import pandas as pd
import preprocessing as pp

best = joblib.load("models/ada_boosting.pkl")

# cross validate the accuracy with logistic regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score

data = pd.read_parquet("RT_IOT2022.parquet")

X_train, X_test, y_train, y_test = pp.preprocess(data)

print(accuracy_score(y_test, best.predict(X_test)))
print(classification_report(y_test, best.predict(X_test)))
print(f"F1 Score: {f1_score(y_test, best.predict(X_test), average='weighted')}")

from sklearn.linear_model import LogisticRegression

log = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
log.fit(X_train, y_train)

joblib.dump(log, "models/logistic_regression.pkl")

# hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

params_grid = { 'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'] }

log = joblib.load("models/logistic_regression.pkl")

search = RandomizedSearchCV(log, params_grid, random_state=42, verbose=3, cv=7, n_jobs=-1, n_iter=10)
search.fit(X_train, y_train)
log = search.best_estimator_

print(accuracy_score(y_test, log.predict(X_test)))
print(classification_report(y_test, log.predict(X_test)))
print(f"F1 Score: {f1_score(y_test, log.predict(X_test), average='weighted')}")


