###IMPORTS
import joblib
import os
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from itertools import product

###


###Store data###
def store_data(X_train_with_A, X_train_without_A, X_val_with_A, X_val_without_A, X_test_with_A, X_test_without_A, y_train, y_val, y_test, age=None, gender=None, education=None,dataset_name='', sufix_name=''):

    if education != None :
        assert len(education) == 3, "There should be a list of train, val test data"
        education_train=education[0]
        education_val=education[1]
        education_test=education[2]
    if age != None :
        assert len(age) == 3, "There should be a list of train, val test data"
        age_train=age[0]
        age_val=age[1]
        age_test=age[2]
    if gender != None :
        assert len(gender) == 3, "There should be a list of train, val test data"
        gender_train=gender[0]
        gender_val=gender[1]
        gender_test=gender[2]



    # Define the directory path for saving dataframes
    dataframes_directory = os.path.join('..', 'Dataframes', dataset_name)

    # Create the 'dataframes' directory if it doesn't exist
    if not os.path.exists(dataframes_directory):
        os.makedirs(dataframes_directory)

    # Save the modified dataframes to CSV files
    X_train_with_A.to_csv(os.path.join(dataframes_directory, f'X_train_with_A{sufix_name}.csv'), index=False)
    X_train_without_A.to_csv(os.path.join(dataframes_directory, f'X_train_without_A{sufix_name}.csv'), index=False)

    X_test_with_A.to_csv(os.path.join(dataframes_directory, f'X_test_with_A{sufix_name}.csv'), index=False)
    X_test_without_A.to_csv(os.path.join(dataframes_directory, f'X_test_without_A{sufix_name}.csv'), index=False)

    X_val_with_A.to_csv(os.path.join(dataframes_directory, f'X_val_with_A{sufix_name}.csv'), index=False)
    X_val_without_A.to_csv(os.path.join(dataframes_directory, f'X_val_without_A{sufix_name}.csv'), index=False)

    y_train.to_csv(os.path.join(dataframes_directory, f'y_train{sufix_name}.csv'), index=False)
    y_test.to_csv(os.path.join(dataframes_directory, f'y_test{sufix_name}.csv'), index=False)
    y_val.to_csv(os.path.join(dataframes_directory, f'y_val{sufix_name}.csv'), index=False)

    # Save the gender and age and education Series to CSV files
    if gender!= None:
        gender_train.to_csv(os.path.join(dataframes_directory, f'gender_train{sufix_name}.csv'), index=False)
        gender_val.to_csv(os.path.join(dataframes_directory, f'gender_val{sufix_name}.csv'), index=False)
        gender_test.to_csv(os.path.join(dataframes_directory, f'gender_test{sufix_name}.csv'), index=False)

    if age != None:
        age_train.to_csv(os.path.join(dataframes_directory, f'age_train{sufix_name}.csv'), index=False)
        age_val.to_csv(os.path.join(dataframes_directory, f'age_val{sufix_name}.csv'), index=False)
        age_test.to_csv(os.path.join(dataframes_directory, f'age_test{sufix_name}.csv'), index=False)

    if education != None:
        education_train.to_csv(os.path.join(dataframes_directory, f'education_train{sufix_name}.csv'), index=False)
        education_val.to_csv(os.path.join(dataframes_directory, f'education_val{sufix_name}.csv'), index=False)
        education_test.to_csv(os.path.join(dataframes_directory, f'education_test{sufix_name}.csv'), index=False)




    print("Dataframes saved in their directory from 'Dataframes' directory.")


def load_stored_data( age=None, gender=None, education=None,dataset_name='', scale=True, without_A=True, sufix_name=''):
    # Define the directory path for the saved dataframes
    dataframes_directory = os.path.join('..', 'Dataframes', dataset_name)

    # Load the dataframes from the CSV files
    X_train_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_train_with_A{sufix_name}.csv'))
    if without_A== True:
        X_train_without_A = pd.read_csv(os.path.join(dataframes_directory, f'X_train_without_A{sufix_name}.csv'))

    X_test_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_test_with_A.csv{sufix_name}'))
    if without_A ==True:
        X_test_without_A = pd.read_csv(os.path.join(dataframes_directory, f'X_test_without_A.csv{sufix_name}'))

    X_val_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_val_with_A.csv{sufix_name}'))
    if without_A ==True:
        X_val_without_A = pd.read_csv(os.path.join(dataframes_directory, f'X_val_without_A{sufix_name}.csv'))
    age_train_,age_val,age_test,gender_train, gender_val, gender_test, ed_train, ed_val,ed_test = None, None, None, None, None, None, None, None, None


    if gender != None:
        gender_train = pd.read_csv(os.path.join(dataframes_directory, f'gender_train{sufix_name}.csv')).values.reshape(-1)
        gender_val = pd.read_csv(os.path.join(dataframes_directory, f'gender_val{sufix_name}.csv')).values.reshape(-1)
        gender_test = pd.read_csv(os.path.join(dataframes_directory, f'gender_test{sufix_name}.csv')).values.reshape(-1)
    if age != None:
        age_train = pd.read_csv(os.path.join(dataframes_directory, f'age_train{sufix_name}.csv')).values.reshape(-1)
        age_val = pd.read_csv(os.path.join(dataframes_directory, f'age_val{sufix_name}.csv')).values.reshape(-1)
        age_test = pd.read_csv(os.path.join(dataframes_directory, f'age_test{sufix_name}.csv')).values.reshape(-1)

    if education != None:
        ed_train = pd.read_csv(os.path.join(dataframes_directory, f'education_train{sufix_name}.csv')).values.reshape(-1)
        ed_val = pd.read_csv(os.path.join(dataframes_directory, f'education_val{sufix_name}.csv')).values.reshape(-1)
        ed_test = pd.read_csv(os.path.join(dataframes_directory, f'education_test{sufix_name}.csv')).values.reshape(-1)

    if scale == True: ############################## TAKE CARE NOT TO SCALE SENS ATTRIBUTES THAT ARE CONSIDERED CLASSES############################
        X_train_with_A, X_val_with_A, X_test_with_A= scale_dataframes(
            [X_train_with_A, X_val_with_A, X_test_with_A]) ###scale all dfs ##Take care scale keeps 0,1 true
        if without_A == True:
            X_test_without_A, X_val_without_A, X_test_without_A=scale_dataframes([X_test_without_A, X_val_without_A,X_test_without_A])




    #age_train_val = np.concatenate((age_train, age_val), axis=0)
    #gender_train_val = np.concatenate((gender_train, gender_val), axis=0)

    # Load target variables (y_train, y_test, y_val)
    y_train = pd.read_csv(os.path.join(dataframes_directory, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(dataframes_directory, 'y_test.csv'))
    y_val = pd.read_csv(os.path.join(dataframes_directory, 'y_val.csv'))

    return X_train_with_A, X_val_with_A, X_test_with_A, y_train, y_val, y_test, age_train,age_val,age_test,gender_train, gender_val, gender_test, ed_train, ed_val, ed_test




def load_data(verbose=False):
    '''
    Return X and y
    '''

    # fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)

    # data (as pandas dataframes)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets
    assert X.shape[0]==1000, "Smth went wrong X should have 1000 rows"
    assert y.shape[0]==1000, 'Smth went wrong y should have 1000 rows'
    print("Data loaded successfully")

    if verbose == True:
        print(f"Variables : {statlog_german_credit_data.variables}")
        print("print X.head()")
        print(X.head(3))
        print()
        print('y.head()')
        print(y.head(3))
    return X, y


import numpy as np

def merge_two_sets(X_train, X_val, y_train, y_val):
    """
    Merge the training and validation sets.

    Parameters:
    - X_train, X_val: Training and validation features
    - y_train, y_val: Training and validation labels

    Returns:
    - X_train_val, y_train_val: Merged training and validation sets
    """
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    return X_train_val, y_train_val

####
def replace_values_with_binary(df, column_name, values_list):
    # Check if the column exists in the dataframe
    assert column_name  in df.columns, "Column name is not correct. It should be in df.column"

    df[column_name] = df[column_name].apply(lambda x: 1 if x in values_list else 0)

    return df

# Sample DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['apple', 'banana', 'cherry', 'apple', 'date']}
df = pd.DataFrame(data)
# Define a list of values to replace
values_list = ['apple', 'banana']
# Test the function
df = replace_values_with_binary(df, 'B', values_list)
# Check if the 'B' column has been modified as expected
assert df['B'].equals(pd.Series([1, 1, 0, 1, 0])), "Function replace_values_with_binary does not work properly"

# The 'B' column should contain [1, 1, 0, 1, 0] after applying the function

###

import pandas as pd

def apply_function_to_column(df, column_name, test_function,new_name):
    """
    Apply the test function to the specified column in the DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame.
    column_name (str): The name of the column to modify.
    test_function (function): The function to apply to each value in the specified column.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    assert column_name in df.columns, "Column name is not correct. It should be in df.columns"

    df[new_name] = df[column_name].apply(test_function)

    return df




from itertools import product

def find_best_model(model, param_grid, X_train, y_train, X_val, y_val):
    """
    Find the best model based on performance on the validation set.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier, SVC, etc.)
    - param_grid: Dictionary with hyperparameter names as keys and lists of hyperparameter values to try
    - X_train, y_train: Training data and labels
    - X_val, y_val: Validation data and labels

    Returns:
    - best_model: The best model trained on the entire training set with the best hyperparameters
    """
    best_score = 0
    best_params = None
    best_model = None

    # Generate all combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))

    for params in param_combinations:
        # Create a dictionary of hyperparameters
        param_dict = dict(zip(param_grid.keys(), params))

        # Set the hyperparameters
        model.set_params(**param_dict)

        # Train the model on the entire training set
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        val_score = model.score(X_val, y_val)

        # Update the best model if the current model has a higher validation score
        if val_score > best_score:
            best_score = val_score
            best_params = param_dict
            best_model = model

    print("Best Model Hyperparameters:", best_params)
    print("Validation Accuracy:", best_score)

    return best_model



def find_best_model_old(model, param_grid, X_train, y_train,verbosee=True):

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    grid_search_accuracy = grid_search.best_score_
    if verbosee== True:
        print(f"Accuracy for best grid search {model} is : {grid_search_accuracy}")
    return best_model, best_params



def eq_op_dif(y_true,y_predicted, sensitive_attribute, no_abs = False):
    """
    Compute Equal Opportunity fairness metric.

    Parameters:
    y_predicted (array-like): Predicted labels (0 or 1).
    y_true (array-like): True labels (0 or 1).
    sensitive_attribute (array-like): Binary sensitive attribute (0 or 1).

    Returns:
    float: Equal Opportunity score (0 to 1).
    """

    # Confusion matrices for different groups (privileged and unprivileged)
    cm_privileged = confusion_matrix(y_true[sensitive_attribute == 1], y_predicted[sensitive_attribute == 1]) ##confusion matrix class 1
    cm_unprivileged = confusion_matrix(y_true[sensitive_attribute == 0], y_predicted[sensitive_attribute == 0]) ## confusion matrix class 0

    # Calculate True Positive Rates (TPR)
    TPR_privileged = cm_privileged[1, 1] / (cm_privileged[1, 0] + cm_privileged[1, 1]) if cm_privileged[1, 1] + cm_privileged[1, 0] > 0 else 0
    TPR_unprivileged = cm_unprivileged[1, 1] / (cm_unprivileged[1, 0] + cm_unprivileged[1, 1]) if cm_unprivileged[1, 1] + cm_unprivileged[1, 0] > 0 else 0

    # Calculate Equal Opportunity score
    equal_opportunity_score = abs(TPR_privileged - TPR_unprivileged)
    if no_abs == True :
        equal_opportunity_score = TPR_privileged - TPR_unprivileged


    return equal_opportunity_score


###Scale the data to 0,1 as in paper

from sklearn.preprocessing import MinMaxScaler


def scale_dataframes(list_of_dfs):
    scaled_dfs = []  # List to store the scaled DataFrames

    for df in list_of_dfs:
        scaler = MinMaxScaler()  # Create a MinMaxScaler object
        scaled_values = scaler.fit_transform(df.values)  # Fit the scaler and transform the values

        # Create a new DataFrame with the scaled values and the same columns and index
        scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

        scaled_dfs.append(scaled_df)  # Append the scaled DataFrame to the list

    return scaled_dfs



####### Use fairness optimizer #########

def use_fairness_optimizer(threshold_optimizer,X_fit, y_fit, X_obs, y_obs, y_train,y_val, sensitive_1_fit, sensitive_2_fit,
                           sensitive_1_obs, sensitive_2_obs, name_1, name_2, fitted=False, name_dataset1 = "train", name_dataset2="val"):

    '''treshold optimiizer should have prefit= True
     sensitive_1 = feature we do optimiziation w.r.t
      sensitive_2 = feature that we see how was the fairness affected
      y_train, y_val = true y
      X_fit, y_fit= like train for fit
      X_obs, y_obs = like validation for optimizer
      y_fit, y_obs= predicted y's before optimizer '''
    
    if fitted == False:
        threshold_optimizer.predict_method = 'auto'
        threshold_optimizer.fit(X_fit, y_train, sensitive_features=sensitive_1_fit)

    adjusted_sensitive_train = threshold_optimizer.predict(X_fit, sensitive_features=sensitive_1_fit)
    adjusted_sensitive_val = threshold_optimizer.predict(X_obs, sensitive_features=sensitive_1_obs)

    print(f"--------- SCORES AFTER OPTIMIZING FOR {name_1} ---------")
    print()
    print("----- accuracy scores -----")

    print(
        f' acc score {name_dataset1} got from : {accuracy_score(y_fit, y_train)} to {accuracy_score(adjusted_sensitive_train, y_train)}')
    print(
        f" acc score {name_dataset2} from : {accuracy_score(y_obs, y_val)} to {accuracy_score(adjusted_sensitive_val, y_val)}")

    print()
    print("----- Scores for fariness -----")

    print(
        f'{name_1} {name_dataset1} eq op went from: {eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_1_fit)} to {eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_1_fit)}')
    print(
        f"{name_1} {name_dataset2} eq op went from :  {eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_1_obs)} to {eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_1_obs)}")

    print(
        f'{name_2} {name_dataset1} eq op went from: {eq_op_dif(y_train, y_fit, sensitive_attribute = sensitive_2_fit, no_abs=True)} to {eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_2_fit, no_abs=True)}')
    print(
        f"{name_2} {name_dataset2} eq op went from :  {eq_op_dif(y_val, y_obs, sensitive_attribute= sensitive_2_obs)} to {eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_2_obs)}")


from sklearn.metrics import accuracy_score

def find_best_threshold( y, y_pred_proba, verbose= True):
    y_scores = y_pred_proba[:, 1]  # Assuming the probability for class 1 is at index 1

    thresholds = np.arange(0, 1, 0.05)  # Threshold values to test
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        accuracy = accuracy_score(y, y_pred)
        accuracies.append(accuracy)

    best_threshold = thresholds[np.argmax(accuracies)]

    if verbose == True:
        plt.plot(thresholds, accuracies, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy at Different Thresholds')
        plt.show()
        print(f"Best treshold is {best_threshold} and best score is {np.max(accuracies)}")

    return best_threshold





################################## functions to add bias ###################
import pandas as pd
import numpy as np


def add_bias(X, y, unprivileged_class_name, unprivileged_class_value, p, verbose=True):
    ''' X = predictors dataframe
        y = values to predict
        unprivileged_class_name = str of unprivileged class
        unprivileged class values = int value for unprivileged (e.g. 0 or 1)
        p=probability to keep '''
    # Identify samples with the unprivileged class
    unprivileged_samples = X[X[unprivileged_class_name] == unprivileged_class_value]  # samples to drop form
    y_indexes = y[y["class"] == 1]
    unprivileged_samples_drop = unprivileged_samples[unprivileged_samples.index.isin(y_indexes.index)]
    if verbose == True:
        print(
            f"Initial positive percentage of samples from unprivileged class is {unprivileged_samples_drop.shape[0] / X.shape[0] * 100} % .")

    # Generate a mask indicating whether each row should be dropped based on the probability p
    drop_mask = np.random.rand(unprivileged_samples_drop.shape[0]) > p

    # Extract the indices of the rows to drop
    samples_to_drop = unprivileged_samples_drop.index[drop_mask].tolist()
    assert len(samples_to_drop) == drop_mask.sum(), "There is an error in process data function"

    # Drop rows from both DataFrames based on the generated mask
    X = X.drop(samples_to_drop)
    y = y.drop(samples_to_drop)
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"

    if verbose == True:
        print(
            f"Current percentage of samples from positive unprivileged class is {(len(drop_mask) - len(samples_to_drop)) / X.shape[0] * 100}% . ")
        print(
            f"No of posiitve samples of unprivileged class kept is {len(drop_mask) - len(samples_to_drop)}, this means {(len(drop_mask) - len(samples_to_drop)) / len(drop_mask) * 100} % . It should be close to {p * len(drop_mask)}")

    return X, y



