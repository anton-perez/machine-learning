import sys
sys.path.append('src')
from matrix import Matrix
from dataframe import DataFrame

class KNearestNeighborsClassifier:
  def __init__(self, k):
    self.k = k 
    self.df = None
    self.dependent_variable = None
  
  def fit(self, df, dependent_variable):
    self.df = df
    self.dependent_variable = dependent_variable

  def compute_distances(self, observation):
    data = self.df.data_dict.copy()
    distances = []
    for i in range(len(data[self.dependent_variable])):
      distance = 0
      for var in observation:
        distance += (observation[var] - data[var][i])**2
      distance = distance**(1/2)
      distances.append(distance)
    
    data['Distance'] = distances
    columns =  ['Distance'] + self.df.columns
    return DataFrame(data, columns).select(['Distance', self.dependent_variable])

  def nearest_neighbors(self, observation):
    return self.compute_distances(observation).order_by('Distance')

  def classify(self, observation):
    k_nearest_neighbors = self.nearest_neighbors(observation).select_rows([i for i in range(self.k)])
    k_nearest_neighbors = k_nearest_neighbors.group_by(self.dependent_variable)
    count_neighbors = k_nearest_neighbors.aggregate('Distance', 'count')
    max_index = count_neighbors.data_dict['Distance'].index(max(count_neighbors.data_dict['Distance']))
    max_value = count_neighbors.data_dict['Distance'][max_index]

    if count_neighbors.data_dict['Distance'].count(max_value) == 1:
      return count_neighbors.data_dict[self.dependent_variable][max_index]
    else:
      avg_neighbors = k_nearest_neighbors.aggregate('Distance', 'avg')
      min_index = avg_neighbors.data_dict['Distance'].index(min(avg_neighbors.data_dict['Distance']))
      return avg_neighbors.data_dict[self.dependent_variable][min_index]
    
    


