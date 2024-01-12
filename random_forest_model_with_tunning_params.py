import numpy as np
import pandas as pd

import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

air_data = pd.read_csv("google_data_analitics\\Invistico_Airline.csv")

# DATA CLEANING
print(air_data.head(10))
print(air_data.value_counts(['satisfaction'])) # this we want to predict/labels (y variable and its categories)

print(air_data.dtypes) # display variable names and types
print(air_data.info())

print(f'Here in the dataset we have {air_data.shape[0]} rows and {air_data.shape[1]} columns.')

print(f'There are {air_data.isna().any(axis=1).sum()} row(s) with missing values.') # check for missing values
print(air_data.isna().sum())

air_cleaned = air_data.dropna(axis=0) # drop missing values
print(f"There are {air_cleaned.isna().any(axis=1).sum()} row(s) with missing values after we've cleaned data.")

# Convert categorical features to one-hot encoded features
air_prepaired = pd.get_dummies(air_cleaned, columns=['Customer Type','Type of Travel','Class'], drop_first=False)
print(air_prepaired.head(10))
print(air_prepaired.dtypes)

# MODEL BUILDING
# Separate the dataset into labels (y) and features (X)
y = air_prepaired['satisfaction']
X = air_prepaired.drop('satisfaction', axis=1)
# Separate the prepared dataset into train and test subsets
X_train_main, X_test, y_train_main, y_test = train_test_split(X, y, 
                                                              test_size=0.25, 
                                                              stratify=y, 
                                                              random_state=42)
# Separate the train subset into train and validation subsets
X_train, X_validate, y_train, y_validate = train_test_split(X_train_main, y_train_main, 
                                                            test_size=0.25, 
                                                            stratify=y_train_main, 
                                                            random_state=42)

# Tune the model
# Determine set of hyperparameters
cv_parameters = {'n_estimators' : [50, 100],
                 'max_depth' : [10, 50],
                 'min_samples_leaf' : [0.5, 1],
                 'min_samples_split' : [0.001, 0.01],
                 'max_features' : ["sqrt"],
                 'max_samples' : [.5,.9]}
# Create list of split indices
split_index = [0 if x in X_validate.index else -1 for x in X_train.index]
custom_split = PredefinedSplit(split_index)
# Instantiate model
random_forest = RandomForestClassifier(random_state=42)
# Using GridSearchCV to search over the specified parameters
random_forest_val = GridSearchCV(estimator=random_forest, param_grid=cv_parameters, 
                                 refit='f1', n_jobs=-1, verbose=1)
# Fit the model
random_forest_val.fit(X_train, y_train)

# Getting/ Obtain optimal parameters
print(random_forest_val.best_params_) # the best parameters for our case

# RESULTS AND EVALUATION
# Using optimal parameters on GridSearchCV
random_forest_optimal = RandomForestClassifier(max_depth=50,
                                               max_features='sqrt',
                                               max_samples=0.9,
                                               min_samples_leaf=1,
                                               min_samples_split=0.001,
                                               n_estimators=100,
                                               random_state=42)
# Fit the optimal model
random_forest_optimal.fit(X_train_main, y_train_main)

# Predict on test set.
y_predicted = random_forest_optimal.predict(X_test)

# Obtain performance scores
# Get precision score
precision_test = precision_score(y_test, y_predicted, pos_label='satisfied')
print(f'The Precision score is {round(precision_test, 3)}.')
# Get recall score
recall_test = recall_score(y_test, y_predicted, pos_label='satisfied')
print(f'The Recall Score is {round(recall_test, 3)}.')
# Get accuracy score
accuracy_test = accuracy_score(y_test, y_predicted)
print(f'The Accuracy score is {round(accuracy_test, 3)}.')
# Get F1 score
f1_score_test = f1_score(y_test, y_predicted, pos_label='satisfied')
print(f'The F1-score is {round(f1_score_test, 3)}.')

# Evaluate the model
# Precision score on test data set
print(f'The precision score is {round(precision_test, 3)} for the test set, which means of all positive predictions {round(precision_test*100, 1)}% prediction are true positive.')
# Recall score on test data set
print(f'The recall score is {round(recall_test, 3)} for the test set, which means of all real positive cases in test set, {round(recall_test *100, 1)}% are predicted positive.')
# Accuracy score on test data set
print(f'The accuracy score is {round(accuracy_test, 3)} for the test set, which means of all cases in test set, {round(accuracy_test*100, 1)}% are predicted true positive or true negative.')
# F1 score on test data set
print(f'The F1 score is {round(f1_score_test, 3)} for the test set, which means the test set has harmonic mean is {round(f1_score_test*100, 1)}%.')

# Evaluate the model
table_ = {'Model': "Tuned Random Forest",
         'F1':  f1_score_test,
         'Recall': recall_test,
         'Precision': precision_test,
         'Accuracy': accuracy_test}

table = pd.DataFrame.from_dict(data=table_, orient='index', columns=[''])
table.transpose()
print(table)
# Or we can do this more easily (easy)
table_duplicate = pd.DataFrame({'Model': 'Tuned Random Forest',
                                'F1': [f1_score_test],
                                'Recall': [recall_test],
                                'Precision': [precision_test],
                                'Accuracy': [accuracy_test]})
print(table_duplicate)

# ILLUSTRATE OUR RESULTS OF BUILDING THE RANDOM FOREST MODEL
conf_matrix = confusion_matrix(y_test, y_predicted, labels=random_forest_optimal.classes_)
display_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=random_forest_optimal.classes_)
display_conf_matrix.plot(values_format='') # `values_format=''` suppresses scientific notation
plt.show()

# SAVE THE RESULT OF MODEL BUILDING
# Define a path to the folder where  we save the model
path = 'D:\\projects\\simple_examples\\google_data_analitics\\'
# Pickle the model
with open(path+'random_forest_cv_model.pickle', 'wb') as to_write:
    pkl.dump(random_forest_val, to_write)
# Open pickled model
with open(path+'random_forest_cv_model.pickle', 'rb') as to_read:
    random_forest_val = pkl.load(to_read)
    print(random_forest_val)
