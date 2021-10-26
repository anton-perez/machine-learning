import pandas as pd
import random
import math
import sys
sys.path.append('src')
from decision_tree import *

class RandomForest:
  def __init__(self, tree_num, min_size_to_split, p):
    self.forest = [DecisionTree(min_size_to_split) for _ in range(tree_num)]
    self.p = p

  def fit(self, data):
    total_len = len(data.index)
    subset_len = math.floor(self.p*total_len)
    for tree in self.forest:
      random_data = data.sample(frac=1).reset_index(drop=True)
      subset = random_data[:subset_len]
      tree.fit(subset)
    
  def predict(self, datapoint):
    predictions = {}
    for tree in self.forest:
      prediction = tree.predict(datapoint)
      if prediction not in predictions:
        predictions[prediction] = 0
      predictions[prediction] += 1
    return max(predictions, key=predictions.get)