"""In a supervised learning scenario, we aim to predict house prices based on features such as square footage, the number of bedrooms, and the neighborhood. We have a dataset where each house is labeled with its corresponding price, making it a classic supervised learning regression problem.

Data
Input features (X): square footage, number of bedrooms, neighborhood, lot size

Target (Y): house price

Solution approach
We can use a regression algorithm, such as linear regression, to map the input features to the target price. The model will be trained using labeled data, where the correct house prices are provided, allowing the model to learn how to predict prices based on the given features.

Steps involved
Data collection: gather historical house data, including features (square footage, etc.) and prices.

Data preprocessing: clean the data, normalize the features, and split it into training and test sets.

Model training: train a linear regression model on the training data.

Model evaluation: use metrics such as MSE to evaluate the accuracy of the model on the test data."""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sample data (square footage, bedrooms, neighborhood as encoded values)
X = [[2000, 3, 1], [1500, 2, 2], [1800, 3, 3], [1200, 2, 1]]
y = [500000, 350000, 450000, 300000]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


"""Outcome
After training, the model can predict house prices for new homes based on the given features. In this case, supervised learning is ideal because we have labeled data and a clear target variable."""