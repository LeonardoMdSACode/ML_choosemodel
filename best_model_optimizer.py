import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump
from datetime import datetime
import os

# read in the CSV file using Pandas
df = pd.read_csv('daily_rental.csv')

print(df)

# Check df for any missing values or other issues
print(df.isnull().sum())

df.info()

X = df[
   ['season',
   'yr',
   'mnth',
   'holiday',
   'weekday',
   'workingday',
   'weathersit',
   'temp',
   'atemp',
   'hum',
   'windspeed']
]

# y = column we want to predict
y = df['rentals']

# convert categorical columns to category data type
X[['season',
   'yr',
   'mnth',
   'holiday',
   'weekday',
   'workingday',
   'weathersit']] = X[['season',
   'yr',
   'mnth',
   'holiday',
   'weekday',
   'workingday',
   'weathersit']].astype('category')

# create dummy variables for categorical columns
X = pd.get_dummies(X, columns=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'], drop_first=True)

def best_performing_model(X, y):
   # split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   scores = {}

   # Linear Regression
   linear = LinearRegression()
   linear.fit(X_train, y_train)
   y_pred = linear.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = linear.score(X_test, y_test)
   print("Linear Regression MSE: ", mse)
   print("Linear Regression R^2: ", r2)
   scores["Linear Regression"] = mse

   # Random Forest
   rng = RandomForestRegressor(n_estimators=100, random_state=0)
   rng.fit(X_train, y_train)
   y_pred = rng.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = rng.score(X_test, y_test)
   print("Random Forest MSE: ", mse)
   print("Random Forest R^2: ", r2)
   scores["Random Forest"] = mse

   # Gradient Boosting
   gboost = GradientBoostingRegressor(random_state=0)
   gboost.fit(X_train, y_train)
   y_pred = gboost.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = gboost.score(X_test, y_test)
   print("Gradient Boosting MSE: ", mse)
   print("Gradient Boosting R^2: ", r2)
   scores["Gradient Boosting"] = mse

   # XGBoost
   xgboost = xgb.XGBRegressor()
   xgboost.fit(X_train, y_train)
   y_pred = xgboost.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = xgboost.score(X_test, y_test)
   print("XGBoost MSE: ", mse)
   print("XGBoost R^2: ", r2)
   scores["XGBoost"] = mse

   # Ridge Regression
   ridge = Ridge(alpha=0.5)
   ridge.fit(X_train, y_train)
   y_pred = ridge.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = ridge.score(X_test, y_test)
   print("Ridge Regression MSE: ", mse)
   print("Ridge Regression R^2: ", r2)
   scores["Ridge Regression"] = mse

   # Lasso Regression
   lasso = Lasso(alpha=0.1)
   lasso.fit(X_train, y_train)
   y_pred = lasso.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = lasso.score(X_test, y_test)
   print("Lasso Regression MSE: ", mse)
   print("Lasso Regression R^2: ", r2)
   scores["Lasso Regression"] = mse

   return scores

scores = best_performing_model(X, y)
print(scores)

choosen_model = min(scores, key=scores.get)
print("Choosen Model: ", choosen_model)

# Gradient Boosting was best model, lets optimize!
def optimize_gradient_boosting(X, y):

   # define the grid of hyperparameters to search
   param_grid = {
      'learning_rate': [0.05, 0.1, 0.2],
      'n_estimators': [50, 100, 200],
      'max_depth': [2, 3, 4],
      'min_samples_split': [2, 3, 4]
   }

   # create a gradient boosting model
   gb_model = GradientBoostingRegressor()

   # perform a grid search with 5-fold cross-validation
   grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error')

   # fit the grid search to the training data
   grid_search.fit(X, y)

   # print the best hyperparameters and model score
   print("Gradient Boost:")
   print("Best parameters:", grid_search.best_params_)
   print("Best MSE score: ", -grid_search.best_score_)
   print("Best R^2 score: ", r2_score(y, grid_search.predict(X)))

optimize_gradient_boosting(X, y)
# Best parameters: {'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 200}
# Best MSE score:  93140.93723823878
# Best R^2 score:  0.988274736819954

# Optimizing 2nd best model, Random Forest
def opt_rand_forest(X, y):
   # Hyperparameter grid
   param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth': [5, 10, 15, 20],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
   }

   # Create Random Forest model
   model = RandomForestRegressor()

   grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

   #Fit the grid search object to the data
   grid_search.fit(X, y)

   print("Random Forest:")
   print("Best Parameters:", grid_search.best_params_)
   print("Best MSE:", -grid_search.best_score_)
   print("Best R^2: ", r2_score(y,grid_search.predict(X)))

opt_rand_forest(X, y)
# Best Parameters: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
# Best MSE: 104934.93291932554
# Best R^2:  0.9564763169663013

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, min_samples_split=2, n_estimators=200)
best_model.fit(X_train, y_train)

# Create predictions for y
y_pred = best_model.predict(X_test)

# Create plots to visually inspect the results
plt.figure(figsize=(10, 6))

# Plot actual vs. predicted values
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Plot distribution of actual values
plt.subplot(2, 2, 2)
plt.hist(y_test, bins=20)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Distribution of Actual Values')
plt.show()

# Plot distribution of predicted values
plt.subplot(2, 2, 3)
plt.hist(y_pred, bins=20)
plt.xlabel('Predicted')
plt.ylabel('Count')
plt.title('Distribution of Predicted Values')
plt.show()

# line plot for the actual values and predictions over time
plt.plot(y_test.values, label="Actual Values")
plt.plot(y_pred, label="Predictions")
plt.xlabel("Time")
plt.ylabel("Bike Rentals")
plt.legend()
plt.show()

# Plot feature importances
feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importances')

plt.tight_layout()
plt.show()

name = "best_model"

# Get the current directory and timestamp
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
dir_path = os.path.dirname(os.path.abspath(__file__))

# Save the model with a timestamp and the given name
file_name = f"{name}_{now}.joblib"
file_path = os.path.join(dir_path, file_name)
dump(best_model, file_path)

# Print the file path where the model is saved
print(f"Model saved at {file_path}")
