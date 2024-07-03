# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load train and test data
train_data = pd.read_csv(r'D:\Prodigy\Task-1\house-prices-advanced-regression-techniques\train.csv')
test_data = pd.read_csv(r'D:\Prodigy\Task-1\house-prices-advanced-regression-techniques\test.csv')

# Display the first few rows of the train and test data
print(train_data.head())
print(test_data.head())

# Define the feature variables (X) and the target variable (y) for training data
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y_train = train_data['SalePrice']

# Define the feature variables (X) for testing data
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Create the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Assuming you have the actual SalePrice for the test data to evaluate the model (if not, skip this step)
if 'SalePrice' in test_data.columns:
    y_test = test_data['SalePrice']
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

# Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': y_pred})
output.to_csv('predictions.csv', index=False)

print(f'Predictions saved to predictions.csv')
