# Data preprocessing

# Importing the libraries
import pandas as pd
import numpy as np

import time
import json

from joblib import dump

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
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

############## GLOBAL VARIABLES ##############
color_red = 'tomato'
color_blue = 'skyblue'
color_green = 'palegreen'

############## GENERIC FUNCTIONS ##############
def encode_categorical_data(data):

    labelencoder = LabelEncoder()
    return labelencoder.fit_transform(data), labelencoder

def one_hot_encode(data, indices):
    
    transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), indices)], remainder = 'passthrough')
    data = transformer.fit_transform(data)

    # Avoiding the Dummy Variable trap
    data = data[:, 1:]
    
    return data

def feature_scaling(data, scaler = None):

    if scaler == None:
        scaler = StandardScaler(with_mean = False)
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    
    return data, scaler

def split_data_set(X, y, test_size = 0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    return X_train, X_test, y_train, y_test

def get_parameter_value(name, parameters, default_value):

    if len(parameters) > 0:
        parameters_dictionary = parameters.split(', ')
        for parameter in parameters_dictionary:
            key_value_pair = parameter.split('=')
            key = key_value_pair[0]
            value = key_value_pair[1]
    
            if(key == name):
                return value

    return default_value

def save_model(path, model):

    return dump(model, path)

def get_model_score(model, X_test, y_test, is_polynomial_regression = False, degree = None):

    if is_polynomial_regression:
        poly_reg = PolynomialFeatures(degree)
        X_test = poly_reg.fit_transform(X_test)
    
    try:
        return model.score(X_test, y_test)
    except TypeError:
        # Avoiding error: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
        return model.score(X_test.toarray(), y_test)

def save_scaler(path, scaler):

    return dump(scaler, path)

def save_label_encoder(path, labelencoder):
    
    return dump(labelencoder, path)
        
def save_regression_plot(scatter_X, scatter_y, plot_X, plot_y, title, xlabel, ylabel, path):
    
    plt.scatter(scatter_X, scatter_y, color = color_red)
    plt.plot(plot_X, plot_y, color = color_blue)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path, bbox_inches = 'tight')
    plt.close()

def save_regression_plot_using_grid(X, y, regressor, labels, title, path):

    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    save_regression_plot(X, y, X_grid, regressor.predict(X_grid), title, labels[0], labels[1], path)
    
def save_classification_plot(X, y, classifier, labels, title, path):
    
    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap((color_red, color_green)))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1], c = ListedColormap((color_red, color_green))(i), label = j)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.savefig(path, bbox_inches = 'tight')
    plt.close()

def get_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    return {
        "true_positive": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "true_negative": int(cm[1, 1]),
        "false_negative": int(cm[1, 0])
    }

############## REGRESSION FUNCTIONS ##############
def fit_linear_regression(X_train, y_train):

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def save_linear_regression_plot(X, y, regressor, labels, title, path):
    
    save_regression_plot(X, y, X, regressor.predict(X), title, labels[0], labels[1], path)
    
def fit_polynomial_regression(X_train, y_train, degree):

    poly_reg = PolynomialFeatures(degree)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)
    return regressor, poly_reg
        
def save_polynomial_regression_plot(X, y, regressor, polynomial_regressor, labels, title, path):

    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    save_regression_plot(X, y, X_grid, regressor.predict(polynomial_regressor.fit_transform(X_grid)), title, labels[0], labels[1], path)

def fit_sv_regression(X_train, y_train, X_test, y_test):

    X_train, scaler_X = feature_scaling(X_train)
    X_test, _ = feature_scaling(X_test, scaler_X)
    
    y_train, scaler_y = feature_scaling(y_train.reshape(-1, 1))
    y_train = y_train.reshape(len(y_train))
    y_test, _ = feature_scaling(y_test.reshape(-1, 1), scaler_y)
    y_test = y_test.reshape(len(y_test))
    
    regressor = SVR(kernel = 'rbf', gamma = 'scale')
    regressor.fit(X_train, y_train)
    return regressor, X_train, y_train, X_test, y_test, scaler_X, scaler_y

def fit_decision_tree_regression(X_train, y_train, criterion):

    regressor = DecisionTreeRegressor(criterion)
    regressor.fit(X_train, y_train)
    return regressor

def fit_random_forest_regression(X_train, y_train, n_estimators, criterion):

    regressor = RandomForestRegressor(n_estimators, criterion)
    regressor.fit(X_train, y_train)
    return regressor

############## CLASSIFICATION FUNCTIONS ##############
def fit_logistic_regression(X_train, y_train, solver):
    
    classifier = LogisticRegression(fit_intercept = True, dual = False, penalty = 'l2', solver = solver)
    classifier.fit(X_train, y_train)
    return classifier

def fit_knn(X_train, y_train, n_neighbors):

    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train)
    return classifier

def fit_svm(X_train, y_train, kernel, gamma):

    classifier = SVC(C = 1.0, kernel = kernel, degree = 3, gamma = gamma, probability = True)
    classifier.fit(X_train, y_train)
    return classifier

def fit_kernel_svm(X_train, y_train, kernel, degree, gamma):

    classifier = SVC(C = 1.0, kernel = kernel, degree = degree, gamma = gamma, probability = True)
    classifier.fit(X_train, y_train)
    return classifier

def fit_naive_bayes(X_train, y_train):

    classifier = GaussianNB()
    try:
        classifier.fit(X_train, y_train)
    except TypeError:
        # Avoiding error: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
        classifier.fit(X_train.toarray(), y_train)
        
    return classifier

def fit_decision_tree_classification(X_train, y_train, criterion, splitter):

    classifier = DecisionTreeClassifier(criterion, splitter)
    classifier.fit(X_train, y_train)
    return classifier

def fit_random_forest_classification(X_train, y_train, n_estimators, criterion):

    classifier = RandomForestClassifier(n_estimators, criterion)
    classifier.fit(X_train, y_train)
    return classifier
    
def process(file_path, features, target, categorical_features, problem_type, algorithm, algorithm_parameters, path, column_names):

    # Converting list of features columns from string to integer
    features = list(map(int, features.split()))
    if len(categorical_features) > 0:
        categorical_features = list(map(int, categorical_features.split()))
        # Find the right index of the column in features list
        categorical_features = np.array(list(map(lambda index: features.index(index), categorical_features)), dtype=int)
    
    # Converting target column from string to integer  
    target = int(target)

    # Importing the dataset
    dataset = pd.read_csv(file_path)
    dataset = dataset.fillna(dataset.mean())
    dataset = dataset.dropna()
    X = dataset.iloc[:, features].values
    y = dataset.iloc[:, target].values

    # Encoding categorical features
    if len(categorical_features) > 0:
        # One hot encoding on X[:, indices]
        X = one_hot_encode(X, categorical_features)
    
    # Encoding categorical target in case of a classification problem
    labelencoder = None
    if problem_type == 'classification':
        y, labelencoder = encode_categorical_data(y)
        
    # Converting column_names to a list of string
    if len(column_names) > 0:
        column_names = column_names.split('::')
        
    # Splitting the dataset into training set and test set
    X_train, X_test, y_train, y_test = split_data_set(X, y)

    now = time.localtime()
    timestamp = time.strftime("%Y%m%d%H%M%S", now)

    if path == None:
        path = 'C:/Temp/'
    model_path = path + 'model' + '_' + timestamp + '.model'
    plot_training_results = ''
    plot_test_results = ''
    scaler_X_path = ''
    scaler_y_path = ''
    labelencoder_path = ''
        
    model = None
    scaler_X = None
    scaler_y = None
    is_polynomial_regression = False
    degree = None
    cmatrix = None
    
    ############## PROBLEM TYPE IS REGRESSION ##############
    if problem_type == 'regression':
    
        # Plots variables
        generate_plots = False
        # Generate plots only if there is 1 dimension
        if X_train.shape[1] == 1:
            generate_plots = True
            plot_training_results = path + 'training_results_' + timestamp + '.png'
            plot_test_results = path + 'test_results_' + timestamp + '.png'
            plot_labels = [ column_names[features[0]], column_names[target] ]
        
        if algorithm == None:
            algorithm = 'linear_regression'
        
        # Selecting the right regression algorithm
        if algorithm == 'linear_regression':
            model = fit_linear_regression(X_train, y_train)
            
            if generate_plots:
                save_linear_regression_plot(X_train, y_train, model, plot_labels, 'Training results', plot_training_results)
                save_linear_regression_plot(X_test, y_test, model, plot_labels, 'Test results', plot_test_results)

        elif algorithm == 'polynomial_regression':
            is_polynomial_regression = True
            degree = int(get_parameter_value('degree', algorithm_parameters, 2))
            model, polynomial_regressor = fit_polynomial_regression(X_train, y_train, degree)
            
            if generate_plots:
                save_polynomial_regression_plot(X_train, y_train, model, polynomial_regressor, plot_labels, 'Training results', plot_training_results)
                save_polynomial_regression_plot(X_test, y_test, model, polynomial_regressor, plot_labels, 'Test results', plot_test_results)

        elif algorithm == 'support_vector_regression':
            model, X_train, y_train, X_test, y_test, scaler_X, scaler_y = fit_sv_regression(X_train, y_train, X_test, y_test)
            
            if generate_plots:
                save_regression_plot_using_grid(X_train, y_train, model, plot_labels, 'Training results', plot_training_results)
                save_regression_plot_using_grid(X_test, y_test, model, plot_labels, 'Test results', plot_test_results)

        elif algorithm == 'decision_tree_regression':
            criterion = get_parameter_value('criterion', algorithm_parameters, 'mse')
            model = fit_decision_tree_regression(X_train, y_train, criterion)
            
            if generate_plots:
                save_regression_plot_using_grid(X_train, y_train, model, plot_labels, 'Training results', plot_training_results)
                save_regression_plot_using_grid(X_test, y_test, model, plot_labels, 'Test results', plot_test_results)

        elif algorithm == 'random_forest_regression':
            criterion = get_parameter_value('criterion', algorithm_parameters, 'mse')
            n_estimators = int(get_parameter_value('n_estimators', algorithm_parameters, 10))
            model = fit_random_forest_regression(X_train, y_train, n_estimators, criterion)
            
            if generate_plots:
                save_regression_plot_using_grid(X_train, y_train, model, plot_labels, 'Training results', plot_training_results)
                save_regression_plot_using_grid(X_test, y_test, model, plot_labels, 'Test results', plot_test_results)
        
    ############## PROBLEM TYPE IS CLASSIFICATION ##############
    elif problem_type == 'classification':
    
        # Feature scaling
        X_train, scaler_X = feature_scaling(X_train)
        X_test, _ = feature_scaling(X_test, scaler_X)
        
        # Plots variables
        generate_plots = False
        # Generate plots only if there is 2 dimensions and 2 classes in target
        if X_train.shape[1] == 2 and len(np.unique(y)) == 2:
            generate_plots = True
            plot_training_results = path + 'training_results_' + timestamp + '.png'
            plot_test_results = path + 'test_results_' + timestamp + '.png'
            plot_labels = [ column_names[features[0]], column_names[features[1]] ]
            
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
            
        # Generate plots
        if generate_plots:
            save_classification_plot(X_train, y_train, model, plot_labels, 'Training results', plot_training_results)
            save_classification_plot(X_test, y_test, model, plot_labels, 'Test results', plot_test_results)
        
        # Generate confusion matrix when target is boolean
        if len(np.unique(y)) == 2:
            cmatrix = get_confusion_matrix(y_test, model.predict(X_test))
    
    # Saving the trained model 
    model_path = save_model(model_path, model)
    
    # Saving the labelencoder
    if labelencoder:
        labelencoder_path = path + 'labelencoder_' + timestamp + '.labelencoder'
        save_label_encoder(labelencoder_path, labelencoder)

    # Saving the scalers
    if scaler_X:
        scaler_X_path = path + 'scaler_X_' + timestamp + '.scaler'
        save_scaler(scaler_X_path, scaler_X)
    if scaler_y:
        scaler_y_path = path + 'scaler_y_' + timestamp + '.scaler'
        save_scaler(scaler_y_path, scaler_y)
    
    # Calculation of model score
    model_score = get_model_score(model, X_test, y_test, is_polynomial_regression, degree)
        
    model_path = ''.join(model_path)
    
    json_object = {
            "model": model_path,
            "scaler_X": scaler_X_path,
            "scaler_y": scaler_y_path,
            "labelencoder": labelencoder_path,
            "score": model_score,
            "confusion_matrix": cmatrix,
            "plot_training_results": plot_training_results,
            "plot_test_results": plot_test_results
    }
    json_string = json.dumps(json_object)

    return json_string

# Main program
if __name__ == '__main__':
    
    # For testing purposes
    file = ''
    column_names = ''
    features = ''
    categorical_features = ''
    target = ''
    
    # classification, regression
    problem_type = ''
    
    # linear_regression, polynomial_regression, support_vector_regression, decision_tree_regression, random_forest_regression
    # logistic_regression, knn, svm, kernel_svm, naive_bayes, decision_tree_classification, random_forest_classification
    algorithm = ''
    algorithm_parameters = ''
    path = ''
    
    result = process(file, features, target, categorical_features, problem_type, algorithm, algorithm_parameters, path, column_names)

