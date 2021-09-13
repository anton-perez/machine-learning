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
  def __init__(self):
    self.data = None #pandas dataframe
    self.root = None
    self.split_vals = None

  def get_split_vals(self, data):
    vars = [var for var in data.columns if var != 'class']
    mid_points = pd.DataFrame(columns = vars)
    for var in vars: 
      var_vals = list(set(data[var]))
      mid_points[var] = [(var_vals[:-1][i]+var_vals[1:][i])/2 for i in range(len(var_vals)-1)]
    return mid_points

  def get_best_splits(self, node):
    data = node.data
    vars = [var for var in data.columns if var != 'class']
    min_entropy = 1
    min_split_var = vars[0]
    min_split_val = self.split_vals[min_split_var][0] 
    min_splits = []
    for var in vars:
      for split_val in self.split_vals[var]:
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
    node.split_var =  min_split_var
    node.children = [Node(data) for data in min_splits]
    print('\nChosen Split: ', node.split_var, '=' ,node.split_val)
    if node.entropy != 0:
      for child in node.children:
        self.get_best_splits(child)

  def fit(self,data):
    self.data = data
    self.root = Node(data)
    self.split_vals = self.get_split_vals(data)
    self.get_best_splits(self.root)

  def calc_entropy(self, data):
    vars = [var for var in data.columns if var != 'class']
    entropy = 0
    total_points = len(data.index) 
    #print(data)
    class_vals = list(set(data['class']))
    for class_val in class_vals:
      p = data[data['class'] == class_val][vars[0]].count() / total_points
      entropy -= p*math.log(p)
    #print('\nEntropy: ', entropy,'\n')
    return entropy 

  def predict(self, datapoint):
    node = self.root
    while node.entropy != 0:
      if datapoint[node.split_var][0] < node.split_val:
        node = node.children[0]
      else: 
        node = node.children[1]
    return list(set(node.data['class']))[0]

      

  