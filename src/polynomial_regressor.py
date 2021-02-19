import sys
sys.path.append('src')
import math
from matrix import Matrix
from dataframe import DataFrame
from linear_regressor import LinearRegressor

class PolynomialRegressor:
  def __init__(self, degree):
    self.degree = degree
    self.dataframe = None
    self.dependent_variable = None
    self.coefficients = None

  def fit(self, df, dependent_variable):
    points = df.to_array()
    point_arr = [[p[0]**e for e in range(1,self.degree+1)]+[p[1]] for p in points]
    indep_var = df.columns[0] if df.columns[0] != dependent_variable else df.columns[1]
    if self.degree > 0:
      columns = [indep_var] + [indep_var + '^' + str(e) for e in range(2, self.degree+1)] + [dependent_variable]
    else:
      columns = [dependent_variable]
    self.dataframe = DataFrame.from_array(point_arr, columns)
    self.dependent_variable = dependent_variable
    self.coefficients = self.calculate_coefficients()
    
  def calculate_coefficients(self):
    return LinearRegressor(self.dataframe, self.dependent_variable).coefficients

  def predict(self, input_dict):
    independent_variables = [var for var in self.dataframe.columns if var != self.dependent_variable]
    sum = self.coefficients['constant']
    for var in independent_variables:
      if var in input_dict:
        sum += self.coefficients[var]*input_dict[var]
      elif '^' in var:
        var_exp = var.split('^')
        sum += self.coefficients[var]*input_dict[var_exp[0]]**int(var_exp[1])
    return sum