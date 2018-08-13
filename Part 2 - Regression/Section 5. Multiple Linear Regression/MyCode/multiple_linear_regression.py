# Importing the libraries
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,3] = labelEncoder_x.fit_transform(x[:,3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
x = oneHotEncoder.fit_transform(x).toarray()

# Avoiding the dummy varible trap
x = x[:,1:]

# Split dataset into training and test
from sklearn.model_selection import train_test_split #train_test_split is deprecated in cross_validation
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to trainning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 

"""
Here plot in graph is not done. 
y_pred is given instead of plotting graph. 
"""
# Prediction of test set results
y_pred = regressor.predict(x_test)

"""
Backward elimination eliminates the colums wihich is not powerful. 
The prediction will be good without these columns.
Here the columns removed whioch is having SL (Significant Level) Value lesser the 0.05(5%). 
SL Value = P>|t| in summary
"""
# Building optimal model using backward elimanation
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() 
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() 
regressor_OLS.summary()
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() 
regressor_OLS.summary()
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() 
regressor_OLS.summary()
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() 
regressor_OLS.summary()

"""
Final result will containig one column that is R&D.
That is the powerful predictor of the profit. 
"""