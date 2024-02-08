import numpy as np
import scipy.stats as st
from math import log2
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin

class EmpiricalDistribution:

    def __init__(self, gaps = None, probobilities = None) -> None:
        self.gaps = gaps
        self.probobilities = probobilities
        self.log_probas = np.log(probobilities)
    
    def pdf(self):
        return self.probobilities[np.digitize(self.gaps)]
    
    def logpdf(self):
        return self.log_probas[np.digitize(self.gaps)]


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self) -> None:
        self.conditional_feature_distributions = {}
        self.prior_label_distibution = {}
        self.unique_labels = {}


    def get_edf(self, arr, bins):
        hist, gaps = np.histogram(arr, bins=bins)
        hist = hist/sum(hist)
        return gaps, hist


    def fit(self, X, y, distibution = None):
        self.unique_labels = np.unique(y)

        self.prior_label_distibution = {
            label: sum((y==label).astype(float)) / len(y)
            for label in self.unique_labels
        }

        for label in self.unique_labels:
            conditional_feature_label_distributions = []
            for feature in range(X.shape[1]):
                feature_column = X[y == label, feature]
                if distibution is None:
                    conditional_feature_label_distributions.append((st.norm(feature_column.mean(), np.var(feature_column))))
                    continue

                gaps, hist = self.get_edf(feature_column, int(log2(feature_column.shape[0]))+1)
                conditional_feature_label_distributions.append(EmpiricalDistribution(gaps, hist))
            self.conditional_feature_distributions[label] = conditional_feature_label_distributions
        

    def predict_log_proba(self, X):
        X_log_proba = np.zeros((X.shape[0], len(self.unique_labels)), dtype=float)
        for label in self.unique_labels:
            for feature in range(X.shape[1]):
                label_feature_pdf = self.conditional_feature_distributions[label][feature].logpdf(X[:, feature])
                X_log_proba[:, label] += label_feature_pdf
            X_log_proba[:, label] += np.log(self.prior_label_distibution[label])
        
        X_log_proba -= logsumexp(X_log_proba, axis=1)[:, None]

        return X_log_proba
    
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        return np.array([self.unique_labels[idx] for idx in log_probas.argmax(axis=1)])


