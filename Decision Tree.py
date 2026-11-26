import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns, na_values="?")

df = df.dropna()

X = df.drop("target", axis=1)
y = (df["target"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("DECISION TREE RESULTS")
print("Training Accuracy:", accuracy_score(y_train, dt.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

print("\nRANDOM FOREST RESULTS")
print("Training Accuracy:", accuracy_score(y_train, rf.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\nBIAS-VARIANCE ANALYSIS")

for depth in [2, 3, 4, 5, 6, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("Depth:", depth)
    print("Training Accuracy:", train_score)
    print("Testing Accuracy:", test_score)
    print("Accuracy Gap:", train_score - test_score)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Decision Tree": dt.feature_importances_,
    "Random Forest": rf.feature_importances_
})

print("\nFEATURE IMPORTANCE")
print(importance.sort_values(
    by="Random Forest", ascending=False
))

importance.set_index("Feature").plot(
    kind="bar", figsize=(12, 6)
)

plt.title("Feature Importance - Decision Tree vs Random Forest")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    confusion_matrix(y_test, dt_pred),
    annot=True, fmt="d", cmap="Blues", ax=axes[0]
)

axes[0].set_title("Decision Tree")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True, fmt="d", cmap="Greens", ax=axes[1]
)

axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()