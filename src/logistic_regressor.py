import sys
sys.path.append('src')
import math
from matrix import Matrix
from dataframe import DataFrame
from linear_regressor import LinearRegressor

class LogisticRegressor:
  def __init__(self, dataframe, dependent_variable):
    self.dataframe = dataframe
    self.dependent_variable = dependent_variable
    self.independent_variables = [var for var in self.dataframe.columns if var != self.dependent_variable]
    self.coefficients = self.calculate_coefficients()
  
  def calculate_coefficients(self):
    df = self.dataframe
    dep_vals = df.data_dict[self.dependent_variable]
    new_dep_vals = [math.log(1/y - 1) for y in dep_vals]
    df.data_dict[self.dependent_variable] = new_dep_vals
    
    return LinearRegressor(df, self.dependent_variable).calculate_coefficients()
  
  def predict(self, input_dict):
    sum = self.coefficients['constant']
    for var in self.independent_variables:
      sum += self.coefficients[var]*input_dict[var]
    sigmoid = 1/(1+math.e**sum)
    return sigmoid