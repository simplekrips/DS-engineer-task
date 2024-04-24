import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load data from a CSV file named 'hidden_test.csv'
df = pd.read_csv('hidden_test.csv')

# Extract feature '6' from the dataframe, convert to numpy array, and reshape for use in prediction
X = df.loc[:, '6'].to_numpy().reshape(-1, 1)

# Load the previously saved PolynomialFeatures object to transform the feature for polynomial regression
# Note: It was not necessary to save this model as it could be redefined in the code.
poly = joblib.load('poly.pkl')
X_poly = poly.transform(X)

# Load the previously trained Linear Regression model
model = joblib.load('model.pkl')

# Use the model to make predictions on the transformed features
y_pred = model.predict(X_poly)

# Add the predicted target values to the original dataframe
df['predicted target'] = y_pred

# Save the dataframe with the predictions to a new CSV file
df.to_csv('prediction.csv', index=False)

# Plotting the predictions: setup the figure size, plot data, set title and labels, and show legend
plt.figure(figsize=(8,5))
plt.scatter(X, y_pred, color='red', label='predicted')
plt.title('Polynomial Regression Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
