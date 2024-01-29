import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

le = LabelEncoder()

def importdata():
    balance_data = pd.read_csv(
        'https://raw.githubusercontent.com/SCANL/datasets/master/ensemble_tagger_training_data/training_data.csv',
        sep=',', header=None)
 
    # Displaying dataset information
    print('Dataset Length: ', len(balance_data))
    print('Dataset Shape: ', balance_data.shape)
    print('Dataset: ', balance_data.head())
    
    label_encoders = {}
    for column in balance_data.columns:
        if balance_data[column].dtype == 'O':
            label_encoder = LabelEncoder()
            balance_data[column] = label_encoder.fit_transform(balance_data[column])
            label_encoders[column] = label_encoder
    
    return balance_data

# Function to split the dataset into features and target variables
def splitdataset(balance_data):

	# Separating the target variable
	X = balance_data.values[:, 2:5]
	Y = balance_data.values[:, 12]

	# Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size=0.3, random_state=100)

	return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train):

	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion='gini',
									random_state=100, max_depth=3, min_samples_leaf=5)

	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini

def train_using_entropy(X_train, X_test, y_train):

	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(
		criterion='entropy', random_state=100,
		max_depth=3, min_samples_leaf=5)

	# Performing training
	clf_entropy.fit(X_train, y_train)
	return clf_entropy

# Function to make predictions
def prediction(X_test, clf_object):
	y_pred = clf_object.predict(X_test)
	print('Predicted values:')
	print(y_pred)
	return y_pred

# Placeholder function for cal_accuracy
def cal_accuracy(y_test, y_pred):
	print('Confusion Matrix: ',
		confusion_matrix(y_test, y_pred))
	print('Accuracy : ',
		accuracy_score(y_test, y_pred)*100)
	print('Report : ',
		classification_report(y_test, y_pred))
 
 # Function to plot the decision tree
def plot_decision_tree(clf_object, feature_names, class_names):
	plt.figure(figsize=(15, 10))
	plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
	plt.show()

if __name__ == '__main__':
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    
    # Visualizing the Decision Trees
    plot_decision_tree(clf_gini, ['X1', 'X2', 'X3', 'X4'], ['N', 'DT', 'CJ', 'P', 'NPL', 'NM', 'V', 'VM', 'PR', 'D', 'PRE'])
    plot_decision_tree(clf_entropy, ['X1', 'X2', 'X3', 'X4'], ['N', 'DT', 'CJ', 'P', 'NPL', 'NM', 'V', 'VM', 'PR', 'D', 'PRE'])

    
    # Evaluating and printing accuracy for both models
    print('Accuracy for Gini model:')
    cal_accuracy(y_test, prediction(X_test, clf_gini))

    print('\nAccuracy for Entropy model:')
    cal_accuracy(y_test, prediction(X_test, clf_entropy))