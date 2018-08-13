# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
# n_estimators is the tree count. Default is 10. 
# 100 Gives 158300 as prediction. 300 Gives 160333 as prediction. 
# 400 Gives 160500 as prediction. So 300 estimators good model.
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x,y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()