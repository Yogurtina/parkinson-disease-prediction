# Import necessary modules
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import utils.common as common

# Tuning done

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

# calculating the accuracy of models with different values of k
mean_acc = np.zeros(20)
for i in range(1,21):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    yhat= knn.predict(X_test)
    mean_acc[i-1] = metrics.accuracy_score(Y_test, yhat)

mean_acc

loc = np.arange(1,21,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,21), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()

knn = KNeighborsClassifier()

grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)

# fit the model on our train set
g_res = gs.fit(X_train, Y_train)

# find the best score
print(g_res.best_score_)

# get the hyperparameters with the best score
bp = g_res.best_params_
print(bp)

# use the best hyperparameters
knn = KNeighborsClassifier(n_neighbors=bp['n_neighbors'],
                           weights=bp['weights'], algorithm='auto',
                           metric = bp['metric'])

knn.fit(X_train, Y_train)

def model_fit(model, X_train, Y_train, X_test, Y_test):
    return model.fit(X_train, Y_train)

common.check_model(knn, model_fit, "KNN Classifier")
