import copy
from math import log
from collections import deque

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class DTNode:
    def __init__(self, x, y, criterion="entropy", parent=None):
        #criterion = entropy | gini
        self.criterion = criterion
        self.x = x
        self.y = y
        self.isleaf = False
        self.parent = parent
        self.devision_thresholds = []
        if(len(x.columns)==0 or len(self.get_y(self.x).unique())==1):
            self.isleaf = True
            self.children = []
            self.target_class = self.get_y(self.x).mode()[0]
        else:
            self.children = []
            selected_feature = self.feature_select()
            self.feature = selected_feature
            features = list(self.x.columns)
            classes = self.classify(self.x ,selected_feature)
            for c in classes:
                dt = c.drop(selected_feature, axis=1)
                self.children.append(DTNode(dt, self.y, self.criterion, parent=self))

    def feature_select(self):
        features = list(self.x.columns)
        l = []
        for f in features:
            if self.criterion == "entropy":
                l.append(self.information_gain(self.x, f))
            elif self.criterion == "gini":
                l.append(self.gini_info_gain(self.x, f))
        l = zip(features, l)
        selected_feature = max(l, key=lambda x:x[1])[0]
        return selected_feature

    def entropy(self, y):
        p_total = y.count()
        e = 0
        for p_i in y.value_counts():
            e += (p_i/p_total)*log((p_i/p_total), 2)
        return -e

    def information_gain(self, dataframe, feature):
        entropy_s = self.entropy(self.get_y(dataframe))
        s_size = len(dataframe)
        classes = self.classify(dataframe, feature)
        ce = 0
        for sv in classes:
            sv_size = len(sv)
            ce+= (sv_size/s_size)*self.entropy(self.get_y(sv))
        igain = entropy_s - ce
        return igain

    def gini(self, y):
        p_total = y.count()
        g = 0
        for p_i in y.value_counts():
            g += (p_i/p_total)**2
        g = 1 - g
        return g

    def gini_info_gain(self, dataframe, feature):
        gini_s = self.gini(self.get_y(dataframe))
        s_size = len(dataframe)
        classes = self.classify(dataframe, feature)
        cg = 0
        for sv in classes:
            sv_size = len(sv)
            cg+= (sv_size/s_size)*self.gini(self.get_y(sv))
        gig = gini_s - cg
        return gig

    def evaluate(self, xt):
        if not self.isleaf:
            if(self.feature in xt.index):
                child_index = self.find_child_index(xt[self.feature])
                xt = xt.drop(self.feature)
                return self.children[child_index].evaluate(xt)
            else:
                print(self.feature)
                print(xt.columns)
                return self.target_class
        else:
            return self.target_class

    def prune(self, validation_x, validation_y):
        #reduced error pruning
        mr_without_pruning = self.calc_node_misclassification_rate(validation_x, validation_y)
        clone = copy.copy(self)
        clone.make_it_leaf()
        mr_with_pruning = clone.calc_node_misclassification_rate(validation_x, validation_y)
        if(mr_without_pruning >= mr_with_pruning):
            self.make_it_leaf()
        for child in self.children:
            child.prune(validation_x, validation_y)

    def make_it_leaf(self):
        self.target_class = self.get_y(self.x).mode()[0]
        self.isleaf = True
        self.children = []

    def calc_node_misclassification_rate(self, validation_x, validation_y):
        err = 0
        for i in validation_x.index:
            if(self.evaluate(validation_x.loc[i]) != validation_y.loc[i]):
                err +=1
        err = err/len(validation_x)
        return err

    def classify(self, dataframe, feature):
        unique_values = sorted(dataframe[feature].unique())
        classes = []
        for v in unique_values:
            classes.append(dataframe[dataframe[feature] == v])
        self.devision_thresholds = []
        for i, v in enumerate(unique_values[0:-1]):
            self.devision_thresholds.append((unique_values[i]+unique_values[i+1])/2)
        return classes

    def find_child_index(self, feature_value):
        for i, v in enumerate(self.devision_thresholds):
            if(feature_value <= v):
                return i
        return len(self.devision_thresholds)

    def get_y(self, x):
        return self.y.loc[x.index]
