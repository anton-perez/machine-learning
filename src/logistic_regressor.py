import sys
sys.path.append('src')
import math
from matrix import Matrix
from dataframe import DataFrame
from linear_regressor import LinearRegressor

class LogisticRegressor:
  def __init__(self, dataframe, dependent_variable, upper_bound):
    self.dataframe = dataframe
    self.dependent_variable = dependent_variable
    self.upper_bound = upper_bound
    self.independent_variables = [var for var in self.dataframe.columns if var != self.dependent_variable]
    self.coefficients = self.calculate_coefficients()
  
  def calculate_coefficients(self):
    df = self.dataframe
    dep_vals = df.data_dict[self.dependent_variable]
    new_dep_vals = [math.log(self.upper_bound/y - 1) for y in dep_vals]
    df.data_dict[self.dependent_variable] = new_dep_vals
    
    return LinearRegressor(df, self.dependent_variable).calculate_coefficients()
  
  def predict(self, input_dict):
    sum = self.coefficients['constant']
    for var in self.independent_variables:
      if var in input_dict:
        sum += self.coefficients[var]*input_dict[var]
      elif ' * ' in var:
        vars = var.split(' * ')
        if vars[0] in input_dict and vars[1] in input_dict:
          sum += self.coefficients[var]*input_dict[vars[0]]*input_dict[vars[1]]
    sigmoid = self.upper_bound/(1+math.exp(sum))
    return sigmoid