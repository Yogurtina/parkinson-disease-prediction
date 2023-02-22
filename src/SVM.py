import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import utils.common as common
#Import svm model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data = pd.read_csv('data.csv', delimiter = ',', header=0)
data = data.drop(labels="name", axis=1)

features = data[['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',
                    'Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA',
                    'NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']].to_numpy()

features = min_max_scaler.fit_transform(features)

labels = data['status'].to_numpy()

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2,random_state=42)

param_grid = {'C':[0.1,1,100,1000],
                'kernel':['rbf','poly','sigmoid','linear'],
                'degree':[1,2,3,4,5,6]}

grid = GridSearchCV(svm.SVC(),param_grid)
grid.fit(X_train,Y_train)

print("params",grid.best_params_)

print(grid.score(X_test,Y_test))

#Predict the response for test dataset
y_pred = grid.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred))

def model_fit(model, X_train, Y_train, X_test, Y_test):
    return model.fit(X_train, Y_train)

common.check_model(grid, model_fit, "SVM Classifier")
