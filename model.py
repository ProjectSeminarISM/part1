# -*- coding: utf-8 -*-
"""
ISM Project Part 1: model.py

Created on Sun Dez 16 11:10:27 2018

@author: Sabljak
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import numpy as np


class Model:

    """Model Selection, Training, Prediction and Evaluation"""
    def __init__(self, f, classifier):
        self.data = f.data
        self.test_data = f.test_data
        self.dev_set_path = f.dev_set_path
        self.train_features = f.train_features
        self.train_classes = f.train_classes
        self.dev_features = f.dev_features
        self.dev_classes = f.dev_classes
        self.test_features = f.test_features
        self.pred_classes = []
        self.prob_classes = []
        self.prob_table = []
        self.classifier = None

        if classifier == 'decision tree':
            self.classifier = DecisionTreeClassifier(max_features=7)
        elif classifier == 'ada-boost':
            self.classifier = AdaBoostClassifier(n_estimators=7)
        elif classifier == 'gaussian':
            self.classifier = GaussianNB()
        elif classifier == 'svc':
            self.classifier = SVC(kernel='rbf', gamma='auto', C=3, probability=True)
        else:
            print('no such classifier. Possible classifiers: "decision tree", "ada-boost", "gaussian", "svc"')

    def train(self):
        self.classifier.fit(self.train_features, self.train_classes)

    def predict(self):
        self.prob_classes = pd.DataFrame(self.classifier.predict_proba(self.dev_features))
        self.prob_classes.columns = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.pred_classes = self.prob_classes.idxmax(axis=1)

    def eval(self):
        conf_mat = confusion_matrix(self.dev_classes, self.pred_classes)
        FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
        FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
        TP = np.diag(conf_mat)
        TN = conf_mat.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        # relevant metrics
        print('accuracy: ', ACC.mean())
        print('sensitivity: ', TPR.mean())
        print('specificity: ', TNR.mean())
        print('F1-score: ', f1_score(self.dev_classes, self.pred_classes, average=None).mean())

        self.prob_table = pd.DataFrame(self.classifier.predict_proba(self.dev_features))
        self.prob_table.columns = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
        self.prob_table['image'] = pd.read_csv(self.dev_set_path)
        self.prob_table = self.prob_table[['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']]
        self.prob_table.to_csv('data/prob_table.csv', index=False)
        # print(self.prob_table.head())

        prob_table_test = pd.DataFrame(self.classifier.predict_proba(self.test_features))
        prob_table_test.columns = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
        prob_table_test['image'] = self.test_data
        prob_table_test = prob_table_test[['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']]
        prob_table_test.to_csv('data/prob_table_testset.csv', index=False)
