import pandas as pd
import random
import math
import sys
sys.path.append('src')
from random_decision_tree import *

class RandomForest:
  def __init__(self, tree_num, min_size_to_split):
    self.forest = [RandomDecisionTree(min_size_to_split) for _ in range(tree_num)]

  def fit(self, data):
    for tree in self.forest:
      tree.fit(data)
    
  def predict(self, datapoint):
    predictions = {}
    for tree in self.forest:
      prediction = tree.predict(datapoint)
      if prediction not in predictions:
        predictions[prediction] = 0
      predictions[prediction] += 1
    return max(predictions, key=predictions.get)


