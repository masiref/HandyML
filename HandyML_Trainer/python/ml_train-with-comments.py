# Data preprocessing

# Importing the libraries
import pandas as pd
import numpy as np

import os 
import time
import json

from joblib import dump
from tempfile import mkdtemp

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
 

def process(file_path, features, target, categorical_features, problem_type, algorithm, algorithm_parameters, path):
    """Converts your data from CSV file to a NumPy arrays.
    
    Parameters
    ----------
    file_path : string
        The file path of your CSV file.
    features : array
        An array of integers corresponding to the features indices in the dataset (zero-based).
    target : integer
        An integer corresponding to the target index in the dataset (zero-based).
    categorical_features : array
        An array of integers corresponding to the categorical features indices in the dataset (zero-based), needed in order to apply encoding.
    problem_type : string
        Must be 'classification' or 'regression' (default = 'regression')
    algorithm : string
        Must be 'linear_regression', 'polynomial_regression', 'support_vector_regression', 'decision_tree_regression', 'random_forest_regression'
        'logistic_regression', 'knn', 'svm', 'kernel_svm', 'naive_bayes', 'decision_tree_classification', 'random_forest_classification'.
    algorithm_parameters : string
        Dictionary of parameters to apply to the algorithm. Must be like 'param1=value1,param2=value2'.
    path : string
        Path of the folder where to save the model.
    Returns
    -------
    X : array

    """

    ############## GENERIC FUNCTIONS ##############
    def encode_categorical_data(data):
        """Encodes categorical data using a LabelEncoder.
    
        Parameters
        ----------
        data : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        data : array-like of shape [n_samples]
        
        """

        labelencoder = LabelEncoder()
        return labelencoder.fit_transform(data)
    
    def one_hot_encode(X, indices):
        """Applying one hot encoding on data.
        
        Parameters
        ----------
        X : array-like of shape [n_samples]
            Target values.
        indices: array
            Columns indices on which to apply one hot encoding
        Returns
        -------
        X : array-like of shape [n_samples]
        
        """
        
        transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), indices)], remainder='passthrough')
        X = np.array(transformer.fit_transform(X), dtype=np.float)
    
        # Avoiding the Dummy Variable trap
        X = X[:, 1:]
        
        return X
    
    def feature_scaling(data, scaler=None):
        """Applying feature scaling on data.
        
        Parameters
        ----------
        data : array-like of shape [n_samples]
            Target values.
        scaler: StandardScaler
            The scaler used. If None, a new StandardScaler is created
        Returns
        -------
        data : array-like of shape [n_samples]
        scaler : StandardScaler
        
        """

        if scaler == None:
            scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    
    def split_data_set(X, y, test_size=0.2):
        """Splitting the dataset into training set and test set.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples, n_targets)
            Target values.
        test_size : float
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        Returns
        -------
        X_train : array-like or sparse matrix, shape (n_samples * (1 - test_size), n_features)
        X_test : array-like or sparse matrix, shape (n_samples * (test_size), n_features)
        y_train : array_like, shape (n_samples * (1 - test_size), n_targets)
        y_test : array_like, shape (n_samples * (test_size), n_targets)
        
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        return X_train, X_test, y_train, y_test

    def get_parameter_value(name, parameters, default_value):
        """Retrieving parameter value by its name, else its default_value.
        
        Parameters
        ----------
        name : string
            Name of the parameter.
        parameters : string
            Dictionary of parameters, should be written like this: 'param1=value1, param2=value2, ...'.
        default_value : object
            Default value of parameter if not found in parameters.
        Returns
        -------
        value : string
        The value of the parameter
        
        """
        
        parameters_dictionary = parameters.split(', ')
        for parameter in parameters_dictionary:
            key_value_pair = parameter.split('=')
            key = key_value_pair[0]
            value = key_value_pair[1]

            if(key == name):
                return value

        return default_value
    
    def save_model(path, model):
        """Saving model on disk for further predictions.
        
        Parameters
        ----------
        path : string
            Path of the folder where to save the model.
        model : Regressor or Classifier
            Model to save.
        Returns
        -------
        model_path : string
        
        """

        return dump(model, path)
    
    def get_model_score(model, X_test, y_test, is_polynomial_regression=False):
        """Returns the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        model : Regressor or Classifier
            Model on which to get the score
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        Returns
        -------
        score : float
        R^2 of self.predict(X) wrt. y in case of Regressor,
        Mean accuracy of self.predict(X) wrt. y in case of Classifier.
        
        """

        if is_polynomial_regression:
            poly_reg = PolynomialFeatures(degree)
            X_test = poly_reg.fit_transform(X_test)

        return model.score(X_test, y_test)
  
    
    ############## REGRESSION FUNCTIONS ##############
    def fit_linear_regression(X_train, y_train):
        """Fitting linear regression to the training set.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        Returns
        -------
        regressor : LinearRegression
        
        """

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        return regressor
    
    def fit_polynomial_regression(X_train, y_train, degree):
        """Fitting polynomial regression to the training set.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        degree : integer
            Polynomial function degree.
        Returns
        -------
        regressor : LinearRegression
        
        """

        poly_reg = PolynomialFeatures(degree)
        X_poly = poly_reg.fit_transform(X_train)
        poly_reg.fit(X_poly, y_train)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y_train)
        return lin_reg_2
        
    
    def fit_sv_regression(X_train, y_train):
        """Fitting support vector regression to the training set.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        Returns
        -------
        regressor : SVR
        
        """

        regressor = SVR()
        regressor.fit(X_train, y_train)
        return regressor
        
    
    def fit_decision_tree_regression(X_train, y_train, criterion):
        """Fitting decision tree regression to the training set.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary.
        criterion : string
            ???
        Returns
        -------
        regressor : DecisionTreeRegressor
        
        """

        regressor = DecisionTreeRegressor(criterion, random_state = 0)
        regressor.fit(X_train, y_train)
        return regressor
    
    def fit_random_forest_regression(X_train, y_train, n_estimators, criterion):
        """Fitting random forest regression to the training set.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary.
        n_estimators : integer
            Number of estimators.
        criterion : string
            ???
        Returns
        -------
        regressor : RandomForestRegressor
        
        """

        regressor = RandomForestRegressor(n_estimators, criterion, random_state = 0)
        regressor.fit(X_train, y_train)
        return regressor

    ############## CLASSIFICATION FUNCTIONS ##############
    def fit_logistic_regression(X_train, y_train, solver):
        """Fitting logistic regression to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary.
        solver : string
            ???
        Returns
        -------
        classifier : LogisticRegressor
        
        """

        classifier = LogisticRegression(fit_intercept=True, dual=False, penalty='l2', solver=solver, random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
       
  
    def fit_knn(X_train, y_train, n_neighbors):
        """Fitting K-Nearest Neighbors to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        n_neighbors : int
            ???
        Returns
        -------
        classifier : KNNClassifier
        
        """

        classifier = KNeighborsClassifier(n_neighbors)
        classifier.fit(X_train, y_train)
        return classifier
    
    def fit_svm(X_train, y_train, kernel, gamma):
        """Fitting SVM to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        kernel : string
            ???
        gamma : float
            ???
        Returns
        -------
        classifier : SVClassifier
        
        """

        classifier = SVC(C=1.0, kernel=kernel, degree=3, gamma=gamma, random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    
    def fit_kernel_svm(X_train, y_train, kernel, degree, gamma):
        """Fitting Kernel SVM to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        kernel : string
            ???
        degree : int
            ???
        gamma : float
            ???
        Returns
        -------
        classifier : KernelSVClassifier
        
        """

        classifier = SVC(C=1.0, kernel=kernel, degree=degree, gamma=gamma, random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    
    def fit_naive_bayes(X_train, y_train):
        """Fitting Naive Bayes to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        Returns
        -------
        classifier : NaiveBayesClassifier
        
        """

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        return classifier
    
    def fit_decision_tree_classification(X_train, y_train, criterion, splitter):
        """Fitting Decision tree classification to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        criterion : string
            ???
        splitter : string
            ???
        Returns
        -------
        classifier : DecisionTreeClassifier
        
        """

        classifier = DecisionTreeClassifier(criterion, splitter, random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    
    def fit_random_forest_classification(X_train, y_train, n_estimators, criterion):
        """Fitting Random forest classification to the training set.
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y_train : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary
        n_estimators : integer
            Number of estimators.
        criterion : string
            ???
        Returns
        -------
        classifier : RandomForestClassifier
        
        """

        classifier = RandomForestClassifier(n_estimators, criterion, random_state = 0)
        classifier.fit(X_train, y_train)
        return classifier
    
    ############## PREPROCESS FUNCTION ##############
    # Converting list of features columns from string to integer
    features = list(map(int, features.split()))
    if len(categorical_features) > 0:
        categorical_features = list(map(int, categorical_features.split()))
    
    # Converting target column from string to integer  
    target = int(target)

    # Importing the dataset
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, features].values
    y = dataset.iloc[:, target].values

    # Encoding categorical features
    if len(categorical_features) > 0:
        # One hot encoding on X[:, indices]
        X = one_hot_encode(X, categorical_features)
        
    # Encoding categorical target in case of a classification problem
    if problem_type == 'classification':
        y = encode_categorical_data(y)
        
    # Splitting the dataset into training set and test set
    X_train, X_test, y_train, y_test = split_data_set(X, y)

    now = time.gmtime()
    now_formatted = time.strftime("%Y%m%d%H%M%S", now)

    if path == None:
        path = 'C:/Temp/'
    model_path = path + algorithm + '_' + now_formatted + '.model'
        
    model = None
    
    ############## PROBLEM TYPE IS REGRESSION ##############
    if problem_type == 'regression':
        if algorithm == None:
            algorithm = 'linear_regression'
        
        # Selecting the right regression algorithm
        if algorithm == 'linear_regression':
            model = fit_linear_regression(X_train, y_train)

        elif algorithm == 'polynomial_regression':
            degree = int(get_parameter_value('degree', algorithm_parameters, 2))
            model = fit_polynomial_regression(X_train, y_train, degree)

        elif algorithm == 'support_vector_regression':
            model = fit_sv_regression(X_train, y_train)

        elif algorithm == 'decision_tree_regression':
            criterion = get_parameter_value('criterion', algorithm_parameters, 'mse')
            model = fit_decision_tree_regression(X_train, y_train, criterion)

        elif algorithm == 'random_forest_regression':
            criterion = get_parameter_value('criterion', algorithm_parameters, 'mse')
            n_estimators = int(get_parameter_value('n_estimators', algorithm_parameters, 10))
            model = fit_random_forest_regression(X_train, y_train, n_estimators, criterion)
        
    ############## PROBLEM TYPE IS CLASSIFICATION ##############
    elif problem_type == 'classification':
        if algorithm == None:
            algorithm = 'logistic_regression'
        
        # Selecting the right classification algorithm
        if algorithm == 'logistic_regression':
            solver = get_parameter_value('solver', algorithm_parameters, 'liblinear')
            model = fit_logistic_regression(X_train, y_train, solver)

        elif algorithm == 'knn':
            n_neighbors = int(get_parameter_value('n_neighbors', algorithm_parameters, 5))
            model = fit_knn(X_train, y_train, n_neighbors)

        elif algorithm == 'svm':
            kernel = get_parameter_value('kernel', algorithm_parameters, 'rbf')
            gamma = float(get_parameter_value('gamma', algorithm_parameters, 0.1))
            model = fit_svm(X_train, y_train, kernel, gamma)

        elif algorithm == 'kernel_svm':
            kernel = get_parameter_value('kernel', algorithm_parameters, 'rbf')
            gamma = float(get_parameter_value('gamma', algorithm_parameters, 0.1))
            degree = int(get_parameter_value('degree', algorithm_parameters, 2))
            model = fit_kernel_svm(X_train, y_train, kernel, degree, gamma)

        elif algorithm == 'naive_bayes':
            model = fit_naive_bayes(X_train, y_train)

        elif algorithm == 'decision_tree_classification':
            criterion = get_parameter_value('criterion', algorithm_parameters, 'entropy')
            splitter = get_parameter_value('splitter', algorithm_parameters, 'best')
            model = fit_decision_tree_classification(X_train, y_train, criterion, splitter)

        elif algorithm == 'random_forest_classification':
            n_estimators = int(get_parameter_value('n_estimators', algorithm_parameters, 10))
            criterion = get_parameter_value('criterion', algorithm_parameters, 'entropy')
            model = fit_random_forest_classification(X_train, y_train, n_estimators, criterion)
    
    # Saving the trained model 
    model_path = save_model(model_path, model)
    
    # Calculation of model score
    is_polynomial_regression = False
    if algorithm == 'polynomial_regression':
        is_polynomial_regression = True
    model_score = get_model_score(model, X_test, y_test, is_polynomial_regression)
        
    model_path = ''.join(model_path)
    
    json_object = { "model": model_path, "score": model_score }
    json_string = json.dumps(json_object)

    # Returning path of the saved model, the score (regression) or accuracy (classification)
    return json_string

# Main program
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data preprocessing for machine learning algorithms based on Scikit-Learn library.')

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-fe', '--features', nargs='+', required=True)
    parser.add_argument('-t', '--target', required=True)
    parser.add_argument('-cfe', '--categorical_features', nargs='+', required=True)
    parser.add_argument('-pt', '--problem_type', choices=['classification', 'regression'], default='regression')
    parser.add_argument('-al', '--algorithm', choices=[
           'linear_regression', 'polynomial_regression', 'support_vector_regression', 'decision_tree_regression', 'random_forest_regression',
           'logistic_regression', 'knn', 'svm', 'kernel_svm', 'naive_bayes', 'decision_tree_classification', 'random_forest_classification'
    ], default='linear_regression')
    parser.add_argument('-alp', '--algorithm_parameters')
    parser.add_argument('-p', '--path', required=True)
    args = parser.parse_args('-f 50_Startups.csv -fe 0 1 2 3 -t 4 -cfe 3 -pt regression -al linear_regression -p C:/Temp/'.split())

    file = args.file
    features = ' '.join(args.features)
    target = args.target
    categorical_features = ' '.join(args.categorical_features)
    problem_type = args.problem_type
    algorithm = args.algorithm
    algorithm_parameters = args.algorithm_parameters
    path = args.path
    
    file = 'data.csv'
    features = '0 1 2 3 4 5 6 7'
    target = '8'
    categorical_features = ''
    
    #           'classification', 'regression'
    problem_type = 'classification'
    
    #            'linear_regression', 'polynomial_regression', 'support_vector_regression', 'decision_tree_regression', 'random_forest_regression',
    #            'logistic_regression', 'knn', 'svm', 'kernel_svm', 'naive_bayes', 'decision_tree_classification', 'random_forest_classification'
    algorithm = 'naive_bayes'
    algorithm_parameters = ''
    path = 'C:/Temp/'
    
    result = process(file, features, target, categorical_features, problem_type, algorithm, algorithm_parameters, path)

