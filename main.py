import random
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import sqlite3

from features import create_features

le = LabelEncoder()


def read_db(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")

    # input_data_copy = input_data.copy()
    # rows = input_data_copy.values.tolist()

    input_data_copy = input_data.copy()
    rows = input_data_copy.values.tolist()
    random.shuffle(rows)

    input_data = pd.DataFrame(rows, columns=input_data.columns)
    return input_data


def import_data(balance_data):
    # Displaying dataset information
    print('Dataset Length: ', len(balance_data))
    print('Dataset Shape: ', balance_data.shape)
    print('Dataset: ', balance_data.head())
    print(balance_data.columns)
    balance_data = create_features(balance_data)

    # Need to pull out the columns we want
    current_features = ['WORD', 'NORMALIZED_POSITION', 'POSITION', 'FIRST_WORD_CAP', 'WORD_LENGTH',
                        'WORD_TOTAL', 'HUNGARIAN']
    df_class = balance_data[['CORRECT_TAG']]
    df_input = balance_data[current_features]

    label_encoders = {}
    for column in df_input.columns:
        if df_input[column].dtype == 'O':
            label_encoder = LabelEncoder()
            df_input[column] = label_encoder.fit_transform(df_input[column])
            label_encoders[column] = label_encoder

    # Return features and correct labels
    return df_input, current_features, df_class


# Function to split the dataset into features and target variables
# df_input is our X (features), df_class is our Y (Correctly labeled samples)
def splitdataset(df_input, df_class):
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df_input, df_class, test_size=0.3, random_state=100)

    return df_input, df_class, X_train, X_test, y_train, y_test


def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion='gini',
                                      random_state=100, max_depth=10)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


def train_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion='entropy', random_state=100, max_depth=10)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred


# Placeholder function for cal_accuracy
def cal_accuracy(y_test, y_pred):
    print('Confusion Matrix: ',
          confusion_matrix(y_test, y_pred))
    print('Accuracy : ',
          accuracy_score(y_test, y_pred) * 100)
    print('Report : ',
          classification_report(y_test, y_pred))


# Function to plot the decision tree
def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()


if __name__ == '__main__':
    conn = sqlite3.connect('scanl_tagger_training_db_1_9_2024.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    sql_query = "SELECT * FROM training_set"
    balance_data = read_db(sql_query, conn)
    df_input, current_features, df_class = import_data(balance_data)

    # Split on features and correct labels
    X, Y, X_train, X_test, y_train, y_test = splitdataset(df_input, df_class)

    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    unique_values = np.unique(y_train)

    params = {
        'max_depth': range(1, 20),
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'min_samples_split': range(2, 10),
        'max_features': range(1, 10)
    }
    stratified_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=100)
    clf = GridSearchCV(DecisionTreeClassifier(random_state=100), params, cv=stratified_kfold)
    clf.fit(X_train, np.ravel(y_train))

    # Best parameters currently (Forest):
    # {'criterion': 'gini', 'max_depth': 17, 'n_estimators': 180}
    # Best Score: 0.9142932776198539

    # Best parameters currently (Tree):
    # {'criterion': 'entropy', 'max_depth': 10, 'max_features': 8, 'min_samples_split': 3, 'splitter': 'best'}
    # Best Score: 0.6758770748982614
    best_model = clf.best_estimator_

    # After grid search completes
    print("Best Parameters:", clf.best_params_)
    print("Best Score:", clf.best_score_)

    # Visualizing the Decision Trees
    plot_decision_tree(clf_gini, current_features, unique_values)
    plot_decision_tree(clf_entropy, current_features, unique_values)

    # Evaluating and printing accuracy for both models
    print('Accuracy for Gini model:')
    cal_accuracy(y_test, prediction(X_test, clf_gini))

    print('\nAccuracy for Entropy model:')
    cal_accuracy(y_test, prediction(X_test, clf_entropy))
