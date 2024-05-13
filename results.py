import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.decomposition import PCA
# from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as pp
import numpy as np
# import sklearn
# from itertools import combinations

data = pd.read_parquet("RT_IOT2022.parquet")
X_train, X_test, y_train, y_test = pp.preprocess(data)

models = []

models.append(joblib.load("models/decision_tree.pkl"))
models.append(joblib.load("models/decision_tree_tuned.pkl"))
models.append(joblib.load("models/gradient_boosting.pkl"))
models.append(joblib.load("models/ada_boosting.pkl"))
models.append(joblib.load("models/logistic_regression.pkl"))

feature_importantces: list = joblib.load("models/feature_importances.pkl")
t10 = feature_importantces[:10]

# # plot the accuracy of each model as a bar graph
# import matplotlib.pyplot as plt
# import seaborn as sns

accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in models]
model_names = [
    "Decision Tree",
    "Tuned Decision Tree",
    "Gradient Boosting",
    "Ada Boosting",
    "Logistic Regression"
]

X = pd.DataFrame(X_train)
# print(X)
y = pd.DataFrame(y_train)

# print(X.columns)
# print(y.columns)

# clf = models[0]

# # Train the model using the top 10 features
# clf.fit(X, y)

# n_classes = 3
# color_palette = plt.cm.coolwarm
# plot_colors = "bwr" # blue, white and red, same as the coolwarm palette
# plot_step = 0.02

# plt.figure(figsize=(25, 12))

# # Generating all pairs of numbers from 0 to 5
# comb = combinations(np.arange(0, 6), 2)

# # Using sets to obtain all unique combinations from 0 to 5 pairs
# unique_combinations = set(comb) 

# # # For each pair of the top 10 features, create a 2D decision boundary plot
# for pair_idx, pair in enumerate(sorted(unique_combinations)):
#     # Only two corresponding features are taken each time
#     X_train_cols = X.to_numpy()[:, pair]

#     # Creating and fitting the classifier to train data
#     classifier = models[0].fit(X_train_cols, y)

#     # Defining a grid of 5 columns and 3 rows 
#     ax = plt.subplot(3, 5, pair_idx + 1)
#     # Plotting the pairs decision boundaries
#     DecisionBoundaryDisplay.from_estimator(classifier,
#                                            X_train_cols,
#                                            cmap=color_palette,
#                                            response_method="predict",
#                                            ax=ax,
#                                            ylabel=X.columns[pair[1]],
#                                            alpha = 0.5)

#     plt.scatter(X_train_cols[:, 0], X_train_cols[:, 1], edgecolor='k')

# plt.suptitle("Decision surface of decision trees trained on pairs of features", fontsize=14)
# plt.legend(loc="lower right")
# plt.show()

from sklearn.tree import plot_tree

# plot the decision tree
plt.figure(figsize=(25, 12))
plot_tree(models[0], filled=True, feature_names=X.columns, class_names=y_train.unique())
plt.savefig("visualizations/decision_tree.png")

