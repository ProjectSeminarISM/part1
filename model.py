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
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import graphviz


class Model:

    """models selection, training, prediction and evaluation"""
    def __init__(self, f, classifier):
        self.data = f.data
        self.train_features = f.train_features
        self.train_classes = f.train_classes
        self.dev_features = f.dev_features
        self.dev_classes = f.dev_classes
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
            # self.classifier = SVC(gamma='auto', C=10, class_weight='balanced', probability=True)
            self.classifier = SVC(gamma='auto', C=30, class_weight='balanced', probability=True)
        else:
            print('no such classifier. Possible classifiers: "decision tree", "ada-boost", "gaussian", "svc"')
            self.classifier = DecisionTreeClassifier()

    def train(self):
        self.classifier.fit(self.train_features, self.train_classes)

        """plot decision_tree
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=f.feature_names,
                                        class_names=f.class_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
    
        graph = graphviz.Source(dot_data)
        graph.view()
        """

    def predict(self):
        self.pred_classes = self.classifier.predict(self.dev_features)
        self.prob_classes = self.classifier.predict_proba(self.dev_features)

    def eval(self):
        # print(confusion_matrix(self.dev_classes, self.pred_classes))
        self.prob_table = pd.DataFrame(self.classifier.predict_proba(self.dev_features), columns=self.classifier.classes_)
        self.prob_table['image'] = pd.read_csv('data/val_split_info.csv')
        self.prob_table = self.prob_table[['image', 'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']]
        self.prob_table.to_csv('data/prob_table.csv', index=False)
        print(self.prob_table.head())
        print(classification_report(self.dev_classes, self.pred_classes))



