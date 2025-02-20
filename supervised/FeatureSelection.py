import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Sample dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Define features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

"""
 Implementing backward elimination
Backward elimination starts with all features and removes those that are not statistically significant based on their p-values.

Steps for backward elimination

Add a constant (intercept) to the feature set.

Fit the model and check p-values.

Remove the feature with the highest p-value greater than 0.05.

Repeat until all remaining features have p-values below 0.05."""

# Add constant to the model
X = sm.add_constant(X)

# Fit the model using Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# Display the model summary
print(model.summary())

# Remove feature with highest p-value if greater than 0.05
if model.pvalues['StudyHours'] > 0.05:
    X = X.drop(columns='StudyHours')
    model = sm.OLS(y, X).fit()

# Final model after backward elimination
print(model.summary())

"""
 Implementing forward selection
Forward selection adds features one at a time based on their contribution to the model’s performance.

Steps for forward selection

Start with an empty model.

Add one feature at a time that improves the model’s performance.

Stop when adding features no longer improves the model."""


def forward_selection(X, y):
    remaining_features = set(X.columns)
    selected_features = []
    current_score = 0.0

    while remaining_features:
        scores_with_candidates = []

        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            scores_with_candidates.append((score, feature))

        # Select the feature with the highest score
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates[0]

        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
        else:
            break

    return selected_features


best_features = forward_selection(X, y)
print(f"Selected features using Forward Selection: {best_features}")


"""Implementing LASSO
LASSO is a regularization technique that automatically shrinks the coefficients of less important features to zero, effectively performing feature selection.

Steps for LASSO

Initialize the LASSO model with a regularization parameter.

Fit the model on the training data.

Analyze which features have nonzero coefficients."""


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LASSO model with alpha (regularization parameter)
lasso_model = Lasso(alpha=0.1)

# Train the LASSO model
lasso_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = lasso_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')

# Display the coefficients of the features
print(f'LASSO Coefficients: {lasso_model.coef_}')