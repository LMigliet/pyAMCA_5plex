import numpy as np
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV


def train_MCA_model(X_MC_train, y_train, max_iter=1000):
    clf_MC = LogisticRegression(max_iter=max_iter) # Create Model
    clf_MC.fit(X_MC_train, y_train) # Train Model
    clf_MC_proba = clf_MC.predict_proba(X_MC_train) # Output Probabilities
    return (clf_MC, clf_MC_proba)
    

def train_ACA_model(X_AC_train, y_train, n_neighbors=20):
    clf_AC = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf_AC.fit(X_AC_train, y_train)
    clf_AC_proba = clf_AC.predict_proba(X_AC_train)
    return (clf_AC, clf_AC_proba)
    

def train_FFI_model(X_FFI_train, y_train, max_iter=1000):
    clf_FFI = LogisticRegression(max_iter=max_iter) # Create Model
    clf_FFI.fit(X_FFI_train, y_train) # Train Model
    clf_FFI_proba = clf_FFI.predict_proba(X_FFI_train) # Output Probabilities
    return (clf_FFI, clf_FFI_proba)
    

def train_AMCA_model(X_AC_MC_train, y_train, max_iter=1000, fit_intercept=False):
#     clf = LogisticRegressionCV(max_iter=max_iter, fit_intercept=fit_intercept, cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0))
    clf = LogisticRegression(max_iter=max_iter, fit_intercept=fit_intercept)
    clf.fit(X_AC_MC_train, y_train)
    clf_proba = clf.predict_proba(X_AC_MC_train)
    return (clf, clf_proba)
