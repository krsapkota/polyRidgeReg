'''
In this example we will train a Polynomal Ridge Regression model on the Boston housing dataset. 
  1. We will first generate train/test split of data.
  2. As a pre-processing step we will then normalize our data to have zero mean and unit variance (along each dimension).
  3. We then specify the parameters to explore (here we only explore regularization coefficient alpha) 
  4. We then fit seperate models for each combination of parameters and validate using 5 fold cross validation.
  5. We finally test the best found model on the test set and quantify the fit.
'''

import numpy as np


# load boston housing data
from sklearn.datasets import load_boston

'''
1. CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town 
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to five Boston employment centres 
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per $10,000 
...
...

'''
(X, y) = load_boston(return_X_y=True)
print X.shape # (506, 13)
print y.shape # (506, )


# train/test (0.8/0.2) split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# data normalization
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train) # scale each dimension to 0 mean and unit variance

X_train_n = scaler.transform(X_train)

# generate polynomial features 
from sklearn.preprocessing import PolynomialFeatures
X_train_n_poly2 = PolynomialFeatures(2).fit_transform(X_train_n) # design matrix 

# create model/estimator instance 
from sklearn.linear_model import Ridge 
clf = Ridge()

# specify parameters of the model/estimator to explore 
from sklearn.model_selection import GridSearchCV
parameters_to_tune = [{'alpha' : [0.1, 1, 10, 100]}]

# grid search on all parameters to be explored and Cross-Validate (CV) to get the best set of parameters
trainer = GridSearchCV(clf, parameters_to_tune, cv=5) # 5 fold cross validation
trainer.fit(X_train_n_poly2, y_train)

# best set of parameters found
print(trainer.best_params_) # best parameter found 
print(trainer.best_score_) # score of the best parameter on the left out fold during CV
print(trainer.best_estimator_) # best estimator 

# test the best model on test split 
X_test_n = scaler.transform(X_test)
X_test_n_poly2 = PolynomialFeatures(2).fit_transform(X_test_n)

# predict
y_pred = trainer.best_estimator_.predict(X_test_n_poly2)
# score prediction 
score =  trainer.best_estimator_.score(X_test_n_poly2, y_test)




