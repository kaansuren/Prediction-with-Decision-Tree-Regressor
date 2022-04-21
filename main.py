# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Csv
data = pd.read_csv("rank_salary.csv")
print(data)

# Slicing
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

# Converting to Numpy arrays
X = x.values
Y = y.values

# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
dt_r = DecisionTreeRegressor(random_state=0)
dt_r.fit(X, Y)

plt.scatter(X, Y)
plt.plot(X, dt_r.predict(X), color = "blue")
plt.show()

print("5.5 level:",dt_r.predict([[5.5]]))