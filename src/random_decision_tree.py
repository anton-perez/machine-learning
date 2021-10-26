import pandas as pd
import math
import random

class Node:
  def __init__(self, data):
    self.data = data
    self.children = []
    self.entropy = 0
    self.split_var = None
    self.split_val = None

class RandomDecisionTree:
  def __init__(self, min_size_to_split):
    self.data = None #pandas dataframe
    self.root = None
    self.min_size_to_split = min_size_to_split

  def get_split_vals(self, data):
    vars = [var for var in data.columns if var != 'class']
    mid_points = {}
    for var in vars: 
      var_vals = list(set(data[var]))
      mid_points[var] = [(var_vals[:-1][i]+var_vals[1:][i])/2 for i in range(len(var_vals)-1)]
    #print(mid_points)
    return mid_points

  def get_random_splits(self, node):
    data = node.data
    vars = [var for var in data.columns if var != 'class']
    node.entropy = self.calc_entropy(data)
    # print(len(data.groupby(vars)))
    # print(pd.DataFrame(data.groupby(vars)))

    if len(data.groupby(vars)) > self.min_size_to_split and node.entropy > 0:
      split_vals = self.get_split_vals(data)
      node.split_var = vars[math.floor(len(vars)*random.random())]
      node.split_val = split_vals[node.split_var][math.floor(len(split_vals[node.split_var])*random.random())]
      splits = [data[data[node.split_var] < node.split_val], data[data[node.split_var] > node.split_val]]
      node.children = [Node(data) for data in splits]
      # print('\nChosen Split: ', node.split_var, '=' ,node.split_val)
      for child in node.children:
        self.get_random_splits(child)

  def fit(self,data):
    self.data = data
    self.root = Node(data)
    #self.split_vals = self.get_split_vals(data)
    self.get_random_splits(self.root)

  def calc_entropy(self, data):
    vars = [var for var in data.columns if var != 'class']
    entropy = 0
    total_points = len(data.index) 
    #print(data)
    class_vals = set(data['class'])
    for class_val in class_vals:
      p = data[data['class'] == class_val][vars[0]].count() / total_points
      entropy -= p*math.log(p)
      
    #print('\nEntropy: ', entropy,'\n')
    return entropy 

  def predict(self, datapoint):
    node = self.root
    while node.children != []:
      if datapoint[node.split_var][0] < node.split_val:
        node = node.children[0]
      else: 
        node = node.children[1]
    return list(node.data['class'].mode())[0]


      

  