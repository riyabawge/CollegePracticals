import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

df = pd.read_csv(url, sep=r"\s+", names=columns, na_values="?")

df = df.dropna()
df = df.drop(columns=["car_name"])

df["origin"] = df["origin"].astype(str)
df = pd.get_dummies(df, columns=["origin"], drop_first=True, dtype=int)

X = df.drop("mpg", axis=1)
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation")
print("----------------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nModel Intercept:", model.intercept_)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients["Absolute Influence"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(
    by="Absolute Influence", ascending=False
)

print("\nFeature Influence:")
print(coefficients.to_string(index=False))

results = pd.DataFrame({
    "Actual MPG": y_test.values,
    "Predicted MPG": y_pred
})

print("\nActual vs Predicted Fuel Efficiency:")
print(results.head(10).to_string(index=False))