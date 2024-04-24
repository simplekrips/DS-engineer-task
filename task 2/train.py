import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load data from CSV file
df = pd.read_csv('train.csv')

# Extract feature '6' and target 'target' from the dataframe, convert to numpy arrays, and reshape for sklearn
X = df.loc[:, '6'].to_numpy().reshape(-1, 1)
y = df.loc[:, 'target'].to_numpy().reshape(-1, 1)

# Split the dataset into training and testing sets with 20% of the data used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features of degree 2 for input feature
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

# Train a linear regression model on the transformed input features
model = LinearRegression()
model.fit(X_poly, y_train)

# Make predictions on the training data
y_pred = model.predict(X_poly)

# Transform the test data using the same polynomial transformer
X_poly_test = poly.transform(X_test)

# Predict the target for the test data
y_pred_test = model.predict(X_poly_test)

# Save the polynomial transformer and model using joblib
joblib.dump(poly, 'poly.pkl')
joblib.dump(model, 'model.pkl')

# Calculate and print R-squared score for the training set
print('Train R^2 score:', r2_score(y_train, y_pred))

# Calculate mean squared error for the training predictions, then compute the root mean squared error
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
print("Train RMSE:", rmse)

# Calculate and print R-squared score for the test set
print('Test R^2 score:', r2_score(y_test, y_pred_test))

# Calculate mean squared error for the test predictions, then compute the root mean squared error
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
print("Test RMSE:", rmse_test)

# Plotting section: display real data points and predictions
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='real')
plt.scatter(X_train, y_train, color='red', label='predicted', s=5)
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
