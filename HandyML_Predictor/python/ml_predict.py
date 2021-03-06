# Importing the libraries
import pandas as pd
import numpy as np

import json

from joblib import load

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from win10toast import ToastNotifier

############## GLOBAL VARIABLES ##############
app_name = 'HandyML Predictor'

############## GENERIC FUNCTIONS ##############
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

def load_sklearn_component(component_path):

    return load(component_path, mmap_mode = None)

############## PREDICT FUNCTIONS ##############
def predict_polynomial (X, model, degree):

    poly_reg = PolynomialFeatures(degree)
    X = poly_reg.fit_transform(X)
    return model.predict(X)
    
def predict_proba(X, model):

    return model.predict_proba(X)
    
def predict(X, model):
    
    try:
        return model.predict(X)
    except TypeError:
        # Avoiding error: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
        return model.predict(X.toarray())

def process(model_path, file_path, features, target, problem_type, algorithm, algorithm_parameters, path, scaler_X_path, scaler_y_path, labelencoder_path, one_hot_encoder_path, dataset_mean_path, dataset_mode_path):

    errors = []
    toaster = ToastNotifier()
    
    model_loaded = False
    
    try:
    
        # Loading the model from disk
        model = load_sklearn_component(model_path)
    
        # Converting list of features columns from string to integer    
        features = list(map(int, features.split()))
    
        # Importing the dataset
        dataset = pd.read_csv(file_path)
        mean = pd.read_pickle(dataset_mean_path)
        dataset = dataset.fillna(mean)
        mode = pd.read_pickle(dataset_mode_path)
        dataset = dataset.fillna(mode)
        X = dataset.iloc[:, features].values
        
        if one_hot_encoder_path:
            one_hot_encoder = load_sklearn_component(one_hot_encoder_path)
            X = one_hot_encoder.transform(X)
    
        if scaler_X_path:
            scaler_X = load_sklearn_component(scaler_X_path)
            X = scaler_X.transform(X)
        
        toaster.show_toast(app_name, 'Model successfully loaded', duration=5)
        model_loaded = True
            
    except Exception as e:
        
        toaster.show_toast(app_name, 'Error raised while loading model', duration=5)
        errors.append('Error raised while loading model: ' + str(e))
        
    y_pred = None
    y_proba = None
    
    if model_loaded:

        try:
            
            # Making the predictions
            if algorithm == 'polynomial_regression':
                degree = int(get_parameter_value('degree', algorithm_parameters, 2))
                y_pred = predict_polynomial(X, model, degree)
        
            else:
                y_pred = predict(X, model)
        
            # Get probability when problem type is classification
            if problem_type == 'classification':
                y_proba = predict_proba(X, model)
                y_proba = np.amax(y_proba, 1)
                y_proba = y_proba * 100
                y_proba = y_proba.tolist()
            
            # Inverse scaling on target if necessary
            if scaler_y_path:
                scaler_y = load_sklearn_component(scaler_y_path)
                y_pred = scaler_y.inverse_transform(y_pred)        
            
            # Inverse label encoding on target if necessary
            if labelencoder_path:
                labelencoder = load_sklearn_component(labelencoder_path)
                y_pred = labelencoder.inverse_transform(y_pred)
                
            y_pred = y_pred.tolist()
        
            toaster.show_toast(app_name, 'Predictions made successfully', duration=5)
        
        except Exception as e:
        
            toaster.show_toast(app_name, 'Error raised while making predictions', duration=5)
            errors.append('Error raised while making predictions: ' + str(e))

    json_object = {
        "errors": errors,
        "y_pred": y_pred,
        "y_proba": y_proba
    }
    json_string = json.dumps(json_object)
    
    return json_string
   
if __name__ == '__main__':
    
    # For testing purposes
    model = ''
    file = ''
    features = ''
    target = ''
    
    # classification, regression
    problem_type = ''
    
    # linear_regression, polynomial_regression, support_vector_regression, decision_tree_regression, random_forest_regression
    # logistic_regression, knn, svm, kernel_svm, naive_bayes, decision_tree_classification, random_forest_classification
    algorithm = ''
    algorithm_parameters = ''

    path = ''

    one_hot_encoder = ''
    scaler_X = ''
    scaler_y = ''
    labelencoder = ''
    dataset_mean = ''
    dataset_mode = ''
    
    result = process(model, file, features, target, problem_type, algorithm, algorithm_parameters, path, scaler_X, scaler_y, labelencoder, one_hot_encoder, dataset_mean, dataset_mode)