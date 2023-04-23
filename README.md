# Finding the best possible model

This project aims to predict the demand for rental bikes using machine learning models. In this project, we use various regression models, such as Linear Regression, Random Forest, Gradient Boosting, XGBoost, Ridge Regression, and Lasso Regression. The best-performing model is chosen based on the mean squared error (MSE) and is then further optimized using grid search to achieve better performance.

### Requirements
- Python 3.7 or higher
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- joblib

### Results
Data requires to predict the rental demand based on multiple factors like weekday, holiday, temperature, etc.
The code works the data, evaluates multiple popular algorithms like Linear regression, Random forest and Gradient Boosting, after picking the most promising models the code will try to optimize the best hyperparameters of the model, show the results of the model and save the model on the current directory with the current timestamp.

With optimized Gradient Boosting it is possible to achieve a R^2 score of 0.989.

![Fig_1](Figure_1.png)

![Fig_4](Figure_4.png)

![Fig_5](Figure_5.png)

(Graphs generated with matplotlib.)

