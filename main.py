from math import log
from collections import deque
from random import shuffle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from graphviz import Digraph

from tree import DTNode, DecisionTree

if __name__ == '__main__':

    df = pd.read_csv("./processed.cleveland.data", header=None,
                names = ['age', 'sex', 'cp', 'trestbps',
                        'chol', 'fbs', 'restecg', 'thalach',
                        'exang', 'oldpeak', 'slope', 'ca',
                        'thal', 'num'])

    #Cleaning and tidying data
    for i in df[df['num']!=0].index:
        df.loc[i, 'num'] = 1
    df = df[(df['ca']!='?')&(df['thal']!='?')]
    df['ca'] = pd.to_numeric(df['ca'])
    df['thal'] = pd.to_numeric(df['thal'])

    numeric_columns = ['age', 'trestbps', 'chol', "thalach", "oldpeak"]
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numeric_features = df[numeric_columns]
    categorical_features = df[categorical_columns]

    def cross_validate(dataframe, k=5):
        indices_list = list(dataframe.index)
        #shuffling data to randomize their order
        shuffle(indices_list)
        ill = len(indices_list)
        fl = int(ill/k)
        validation_indices = [indices_list[i*fl:(i+1)*fl] for i in range(0, k)]
        def cmr_decorater(func):
            def wrapper(criterion):
                results = []
                for i in range(0, k):
                    training_indices = [x for x in indices_list if x not in validation_indices[i]]
                    tree = func(dataframe.loc[training_indices, :], dataframe.loc[validation_indices[i], :], criterion)
                    results.append(tree)
                return results
            return wrapper
        return cmr_decorater

    @cross_validate(df, 5)
    def calc_misclassification_rate(training_dataframe, validation_dataframe, criterion):
        err = 0
        x = training_dataframe[categorical_columns]
        y = training_dataframe['num']
        dt = DecisionTree(criterion)
        dt.fit(x, y)
        dt.prune(validation_dataframe.loc[:, validation_dataframe.columns != "num"], validation_dataframe.loc[:, "num"])
        for i in validation_dataframe.index:
            if(dt.root.evaluate(validation_dataframe.loc[i, validation_dataframe.columns != "num"]) != validation_dataframe.loc[i, "num"]):
                err +=1
        err = err/len(validation_dataframe)
        print((err, dt))
        return (err, dt)

        gini_trees = calc_misclassification_rate(criterion="gini")
        gtree = max(gini_trees, key=lambda x:x[0])[1]
        print("best gini tree = {}".format(gtree))
        Gg = Digraph("", filename="tree_gini.pdf")
        gtree.plot(Gg)
        Gg.view()
        entropy_trees = calc_misclassification_rate(criterion="entropy")
        etree = max(entropy_trees, key=lambda x:x[0])[1]
        print("best entropy tree = {}".format(etree))
        Ge = Digraph("", filename="tree_entropy.pdf")
        etree.plot(Ge)
        Ge.view()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(categorical_features, df.num)
        tree.plot_tree(clf, ax=ax)
        plt.savefig("sklearn_entropy")
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        clf = tree.DecisionTreeClassifier(criterion="gini")
        clf = clf.fit(categorical_features, df.num)
        tree.plot_tree(clf, ax=ax)
        plt.savefig("sklearn_gini")
        plt.show()
