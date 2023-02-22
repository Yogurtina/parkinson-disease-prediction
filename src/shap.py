from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras.backend as K
# import kerastuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)

# F1 Score
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def get_data():

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    data = pd.read_csv('data.csv', delimiter = ',', header=0)
    data = data.drop(labels="name", axis=1)

    labels = data['status'].to_numpy()
    data = data.drop(labels="status", axis=1)

    features = data[data.columns.to_numpy()]
    features = min_max_scaler.fit_transform(features)

    # split into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)
    # return X_train, X_test, Y_train, Y_test

# Leave One Group Out Cross Validation

    samples = 6 
    patients = len(features)//samples
    groups = [ [i] * samples for i in range(patients)]
    groups = np.array(groups).flatten()

    logo = LeaveOneGroupOut()

    for i, (train_index, test_index) in enumerate(logo.split(features, labels, groups)):

        group_X_train = features[train_index]
        group_X_test = features[test_index]
        group_Y_train = labels[train_index]
        group_Y_test = labels[test_index]

        yield group_X_train, group_X_test, group_Y_train, group_Y_test

""" 
for i, (group_X_train, group_X_test, group_Y_train, group_Y_test) in enumerate(get_data()):
    print(f"Group {i}")
    model.fit(group_X_train, group_Y_train, epochs=10, verbose=0)
"""

# Check model

def check_model_tensorflow(model, model_fit_func, name):
    all_metrics = ['binary_accuracy',
           tf.keras.metrics.Accuracy(),
           tf.keras.metrics.SpecificityAtSensitivity(0.0),
           tf.keras.metrics.SensitivityAtSpecificity(0.0),
           tf.keras.metrics.AUC(),
           BalancedSparseCategoricalAccuracy(),
           get_f1, ]


    all_metrics = pd.DataFrame()

# make the dataframe have as column names the model's metrics
    for metric in model.metrics_names:
        all_metrics[metric] = []

    for i, (group_X_train, group_X_test, group_Y_train, group_Y_test) in enumerate(get_data()):
        print(f"Group {i}")
        history = model_fit_func(model, group_X_train, group_Y_train, group_X_test, group_Y_test)
        scores = model.evaluate(group_X_test, group_Y_test, verbose=0)

        # add the model's validation metrics to the dataframe
        for metric in model.metrics_names:
            all_metrics.loc[i, metric] = scores[model.metrics_names.index(metric)]

    metrics_mean = all_metrics.mean()
    metrics_std = all_metrics.std()

    for metric in model.metrics_names:
        print(f"{metric}: {metrics_mean[metric]:.3f} +/- {metrics_std[metric]:.3f}")

def specificity_metric(y_true, y_pred):
    tp, fp, tn, fn = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp)

def sensitivity_metric(y_true, y_pred):
    tp, fp, tn, fn = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn)

def check_model(model, model_fit_func, name):

    all_metrics = [
        metrics.accuracy_score,
        metrics.f1_score,
        metrics.auc,
        metrics.balanced_accuracy_score,
        specificity_metric,
        sensitivity_metric,
    ]

    metric_scores = pd.DataFrame()

# make the dataframe have as column names the model's metrics
    for metric in all_metrics:
        metric_scores[metric.__name__] = []

    for i, (group_X_train, group_X_test, group_Y_train, group_Y_test) in enumerate(get_data()):
        print(f"Group {i}")
        history = model_fit_func(model, group_X_train, group_Y_train, group_X_test, group_Y_test)

        # scores = model.evaluate(group_X_test, group_Y_test, verbose=0)

        # add the model's validation metrics to the dataframe
        for metric in all_metrics:
            metric_scores.loc[i, metric.__name__] = metric(group_Y_test, model.predict(group_X_test))

        metric_scores_mean = metric_scores.mean()
        metric_scores_std = metric_scores.std()

    for metric in all_metrics:
        print(f"{metric.__name__}: {metric_scores_mean[metric.__name__]:.3f} +/- {metric_scores_std[metric.__name__]:.3f}")



data = pd.read_csv('data.csv', delimiter = ',', header=0)
data = data.drop(labels="name", axis=1)

features = data[['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',
                    'Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA',
                    'NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']].to_numpy()

labels = data['status'].to_numpy()

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2,random_state=42)

n_estimators = [5,20,50,100] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap}


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
rf_random.fit(X_train, Y_train)

print ('Random grid: ', random_grid, '\n')
# print the best parameters
print ('Best Parameters: ', rf_random.best_params_, ' \n')

# apply best hyperparameters
randmf = RandomForestClassifier(n_estimators = 50, min_samples_split = 6,
     min_samples_leaf= 1, max_features = 'sqrt', max_depth= 40, bootstrap=False) 

randmf.fit(X_train, Y_train) 

#Predict the response for test dataset
y_pred = randmf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred))

def model_fit(model, X_train, Y_train, X_test, Y_test):
    return model.fit(X_train, Y_train)

check_model(randmf, model_fit, "Random Forest Classifier")

# import shap library
import shap

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(randmf)
shap_values = explainer.shap_values(X_train)

shap.initjs()
# shap.summary_plot(shap_values, X_train, plot_type="bar")

shap.force_plot(explainer.expected_value[1], shap_values[1], X_train)


# shap.summary_plot(shap_values, X_train)