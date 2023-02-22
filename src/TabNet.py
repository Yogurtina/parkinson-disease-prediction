import numpy  as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
import utils.common as common

model = TabNetClassifier(verbose=0,seed=42)

def model_fit(model, X_train, y_train, X_test, y_test):
    return model.fit(X_train=X_train, y_train=y_train,
               patience=5,max_epochs=500,
               eval_metric=['auc'])

common.check_model(model, model_fit, "TabNet Classifier")
