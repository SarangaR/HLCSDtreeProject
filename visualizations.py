import pandas as pd

import matplotlib.pyplot as plt
import joblib

# Load the data
data = pd.read_parquet("RT_IOT2022.parquet")

# Find the most important features to predict the target variable
import seaborn as sns
import os
import os
data = data.dropna()
data = data.drop(["Unnamed: 0"], axis=1)

X = data.drop("Attack_type", axis=1)
categoricals = X.select_dtypes(include=[object]).columns.to_list()

y = data["Attack_type"]

# # # one hot encoding
X = pd.get_dummies(X, columns=categoricals, drop_first=True)

def summary():
    print(data.describe())
    print(data.describe().T)

summary()

# # # # Find the most important features to predict the target variable
cols = categoricals
cols = [col for col in cols if col != "Attack_type"]

fixed_data = pd.get_dummies(data, columns=cols, drop_first=True)


# print(fixed_data.columns)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

joblib.dump(feature_importances, "models/feature_importances.pkl")

print(feature_importances[:10])

# # sort the data_short columns by importance
f = fixed_data[feature_importances.index]

data_short = f.copy()
for col in data_short.columns:
    if len(col) > 5:
        data_short = data_short.rename(columns={col: col[:5]})

fig = data_short.hist(figsize=(20, 20))
[x.title.set_size(32) for x in fig.ravel()]
plt.savefig("visualizations/hist.png")

fig = data_short.hist(figsize=(20, 20))
[x.title.set_size(32) for x in fig.ravel()]
plt.savefig("visualizations/hist.png")


# # # Plot the 10 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['importance'][:10], y=feature_importances.index[:10])
plt.title("Top 10 most important features")
plt.savefig("visualizations/important_features.png")
plt.clf()

# # Plot the distribution of the target variable
plt.figure(figsize=(20, 20))
# data.hist()
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("visualizations/hist.png")
plt.clf()

# # Plot the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(pd.get_dummies(data).corr())
plt.title("Correlation Matrix")
plt.savefig("visualizations/correlation_matrix.png")
plt.clf()

plt.figure(figsize=(25, 6))
sns.scatterplot(x="Attack_type", y="id.resp_p", data=data)
plt.title("id.resp_p vs Attack_type")
plt.savefig("visualizations/id_resp_p_vs_attack_type.png")
plt.clf()

fig = plt.figure(figsize=(25, 6))
for i, feature in enumerate(feature_importances.index[:10]):
    plt.subplot(2, 5, i+1)
    sns.scatterplot(x=feature, y="Attack_type", data=data)
    plt.title(f"{feature} vs Attack_type")
plt.tight_layout()
plt.savefig("visualizations/important_features_vs_attack_type.png")

from sklearn.tree import DecisionTreeClassifier
clf: DecisionTreeClassifier = joblib.load("models/decision_tree_tuned.pkl")
print(clf.get_params())

