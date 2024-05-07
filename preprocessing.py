import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def preprocess(data : pd.DataFrame):
    data = data.dropna()
    data = data.drop(["Unnamed: 0"], axis=1)

    X = data.drop(["Attack_type"], axis=1)
    categoricals = X.select_dtypes(include=[object]).columns.to_list()

    y = data["Attack_type"]

    # # # one hot encoding
    X = pd.get_dummies(X, columns=categoricals, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test