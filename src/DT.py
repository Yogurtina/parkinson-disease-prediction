from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import utils.common as common
from sklearn import metrics

# read file
data = pd.read_csv('data.csv', delimiter = ',', header=0)
data = data.drop(labels="name", axis=1)

# feature selection
features = data[['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',
                    'Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA',
                    'NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']].to_numpy()

labels = data['status'].to_numpy()

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2,random_state=50)

data = data.drop(labels="status", axis=1)


# DT
dt = tree.DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
# Tuning
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train, Y_train)

# get the hyperparameters with the best score
bp = grid_search.best_params_
print(bp)

#Predict the response for test dataset
y_pred = grid_search.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# # Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred))

# # Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred))


def model_fit(model, X_train, Y_train, X_test, Y_test):
    return model.fit(X_train, Y_train)

common.check_model(grid_search, model_fit, "Decision Tree Classifier")
