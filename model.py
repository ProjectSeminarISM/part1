# -*- coding: utf-8 -*-
"""
ISM Project Part 1: model.py

Created on Sun Dez 16 11:10:27 2018

@author: Sabljak
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import graphviz


class Model:

    """models selection, training, prediction and evaluation"""
    def __init__(self, f, classifier):
        self.train_features = f.train_features
        self.train_classes = f.train_classes
        self.dev_features = f.dev_features
        self.dev_classes = f.dev_classes
        self.pred_classes = []
        self.classifier = None

        if classifier == 'decision tree':
            self.classifier = DecisionTreeClassifier()
        elif classifier == 'ada-boost':
            self.classifier = AdaBoostClassifier()
        elif classifier == 'gaussian':
            self.classifier = GaussianNB()
        elif classifier == 'svc':
            self.classifier = SVC(gamma=2, C=0.1, class_weight='balanced', probability=True)
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

    def eval(self):
        # print(confusion_matrix(self.dev_classes, self.pred_classes))
        print(classification_report(self.dev_classes, self.pred_classes))

