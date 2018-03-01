# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#We dont need a position the Categorical variableTo construct matrix we are giving the range of level instead of index of level
x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

# Fit linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Visualizing the linear regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

""" Code optimised below.
# Fit linear polynomial to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
 
# Visualizing the polynomial regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
"""

# Fit linear polynomial to the dataset. For more accurate changing(increasing) the degree
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y)

# Visualizing the polynomial regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg3.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

""" Below code is used to draw curve with more x values
# Visualizing the polynomial regression results
# Below we drew prediction curve on 1,1.1,1.2...10
# Before it was 1,2,3...10
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg3.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
"""

# Predicting The Resluts
# Linear regression
lin_reg.predict(6.5)
# polynomial regression
lin_reg3.predict(poly_reg.fit_transform(6.5))