# gdlearn - Machine learning module
-Made with ❤️ by Godly.

The purpose of such a module is just to add some fun flavours to the process of logic building and understanding the crux of the machine algorithms.

Learning with implementation boosts the understanding 💯💯.

This module is created by using Python programming language and also some of the famous libraries like pandas(for dataframe) and numpy(for list and array).

To use this module you can add the gdlearn.py file to the site-packages folder in python.

There are four supervised machine learning algorithms in this module:
  1. Linear Regression.
  2. Logistic Regression.
  3. KNN Classifier.
  4. Naive Bayes.
  
## 1. Linear Regression:

Linear regression is all about fitting up the line to the data.

    - Import: from gdlearn import SLinearRegression()
    - Creating a model: model = SLinearRegression()
    - Training the model: model.fit(X,Y)
      - this trains the model based on X and Y.
    - Prediction: model.predict(nx)
        model predicts for the new value of X (nx).
    - Testing the model: model.predict(test_X,test_Y,method = ['MSE','RMSE','MAE'])
        model tests for the values of test_X and compares its output with the original output test_Y.
  
## 2. Logistic Regression:

Logistic regression is a classification algorithm.

    - Import: from gdlearn import LogisticRegression()
    - Creating a model: model = LogisticRegression()
    - Training the model: model.fit(X,Y)
      - this trains the model based on X and Y.
    - Prediction: model.predict(nx, threshold)
        model predicts for the new value of X (nx).
        threshold is the value of the decision boundary.
    - Testing the model: model.predict(test_X,test_Y,threshold)
        model tests for the values of test_X and compares its output with the original output test_Y.
        
## 3. KNN Classifier:

K-Nearest Neighbours(KNN) Classifier is a classification algorithm. Also called as lazy learner.

    - Import: from gdlearn import KNNClassifier()
    - Creating a model: model = KNNClassifier()
    - Training the model: model.fit(X,Y)
      - this trains the model based on X and Y.
    - Prediction: model.predict(lst, k)
        lst is the list of values of X.
        k is the number of neighbouring data points the model should compare with.
    - Testing the model: model.predict(test_X,test_Y,k)
        model tests for the values of test_X and compares its output with the original output test_Y.

## 4. Naive Bayes:

The Naive Bayes classification algorithm is a probabilistic classifier. 
It is based on probability models that incorporate strong independence assumptions.

    - Import: from gdlearn import NaiveBayes()
    - Creating a model: model = NaiveBayes()
    - Training the model: model.fit(X,Y)
      - this trains the model based on X and Y.
    - Prediction: model.predict(X)
    - Testing the model: model.predict(test_X,test_Y)
        model tests for the values of test_X and compares its output with the original output test_Y.
        
        
        
The module gdlearn is still under construction. Hope to add new algorithms very soon.

Thank You 🤝🤝.