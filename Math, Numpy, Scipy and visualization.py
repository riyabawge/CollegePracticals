import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("MATH")
x = 25
print("Square Root:", math.sqrt(x))
print("Power:", math.pow(2, 5))
print("Factorial:", math.factorial(5))
print("Ceiling:", math.ceil(4.3))
print("Floor:", math.floor(4.7))

print("\nNUMPY")
marks = np.array([65, 72, 80, 75, 90])
print("Marks:", marks)
print("Sum:", np.sum(marks))
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard Deviation:", np.std(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))

print("\nSCIPY")
print("Z-Scores:", stats.zscore(marks))
mode = stats.mode(marks, keepdims=True)
print("Mode:", mode.mode[0])

print("\nPANDAS")
data = {
    "Student": ["Amit", "Neha", "Rahul", "Sneha", "Priya"],
    "Maths": [78, 85, 72, 90, 88],
    "Python": [82, 90, 75, 95, 86],
    "AI": [80, 88, 70, 92, 90]
}

df = pd.DataFrame(data)
print(df)

df["Average"] = df[["Maths", "Python", "AI"]].mean(axis=1)
print(df)

plt.bar(df["Student"], df["Average"])
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Average Marks")
plt.show()

sns.scatterplot(x="Maths", y="Python", data=df)
plt.title("Maths vs Python")
plt.show()

correlation = df[["Maths", "Python", "AI"]].corr()
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()