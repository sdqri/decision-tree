from math import log
from collections import deque

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .dtnode import DTNode

class DecisionTree:

    def __init__(self, criterion="entropy"):
        #criterion = entropy | gini
        self.criterion = criterion
        self.root = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.root = DTNode(x, y, criterion=self.criterion)

    def prune(self, validation_x, validation_y):
        self.root.prune(validation_x, validation_y)

    def plot(self, graph):
        q = deque()
        q.append(self.root)
        while(q):
            node = q.pop()
            if(node.isleaf):
                graph.node(str(id(node)), str(node.target_class))
            else:
                graph.node(str(id(node)), str(node.feature) + "\n" + str(node.devision_thresholds))
            for child in node.children:
                graph.edge(str(id(node)), str(id(child)))
                q.append(child)
