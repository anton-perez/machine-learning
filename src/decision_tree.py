import pandas as pd
import math

class Node:
  def __init__(self, data):
    self.data = data
    self.children = []
    self.entropy = 0
    self.split_var = None
    self.split_val = None

class DecisionTree:
  def __init__(self, min_size_to_split):
    self.data = None #pandas dataframe
    self.root = None
    self.min_size_to_split = min_size_to_split

  def get_split_vals(self, data):
    vars = [var for var in data.columns if var != 'class']
    midpoints = {}
    for var in vars: 
      var_vals = list(set(data[var]))
      midpoints[var] = [(var_vals[:-1][i]+var_vals[1:][i])/2 for i in range(len(var_vals)-1)]
    return midpoints

  def get_best_splits(self, node):
    data = node.data
    split_vals = self.get_split_vals(data)
    vars = [var for var in data.columns if var != 'class']
    
    if len(data.groupby(vars)) > self.min_size_to_split and self.calc_entropy(data) != 0:
      #print(pd.DataFrame(data.groupby(vars)))
      min_entropy = 1
      i = 0
      min_split_var = vars[i] 
      while split_vals[min_split_var] == []:
        i+=1
        min_split_var = vars[i] 
      min_split_val = split_vals[min_split_var][0] 
      min_splits = []
      for var in vars:
        for split_val in split_vals[var]:
          df1 = data[data[var] < split_val]
          df2 = data[data[var] >= split_val]
          entropy1 = self.calc_entropy(df1)
          entropy2 = self.calc_entropy(df2)
          total_entropy = (len(df1.index)*entropy1+len(df2.index)*entropy2)/len(data.index)
          if total_entropy < min_entropy:
            min_entropy = total_entropy
            min_split_val = split_val
            min_split_var = var
            min_splits = [df1,df2]
      node.entropy = min_entropy
      node.split_val = min_split_val
      node.split_var = min_split_var
      node.children = [Node(data) for data in min_splits]
      print('\nChosen Split: ', node.split_var, '=' ,node.split_val)
      if node.entropy != 0:
        for child in node.children:
          self.get_best_splits(child)

  def fit(self,data):
    self.data = data
    self.root = Node(data)
    self.get_best_splits(self.root)

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

      

  