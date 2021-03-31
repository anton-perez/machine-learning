import sys
sys.path.append('src')
import math
from matrix import Matrix
from dataframe import DataFrame
from linear_regressor import LinearRegressor

class LogisticRegressor:
  def __init__(self, dataframe, dependent_variable, upper_bound=1):
    self.dataframe = dataframe
    self.dependent_variable = dependent_variable
    self.upper_bound = upper_bound
    self.independent_variables = [var for var in self.dataframe.columns if var != self.dependent_variable]
    try:
      self.coefficients = self.calculate_coefficients()
    except:
      self.coefficients = {key:0 for key in ['constant']+self.independent_variables}
  
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

  def copy(self):
    return LogisticRegressor(self.dataframe, self.dependent_variable, self.upper_bound)

  def calc_rss(self):
    rss = 0
    for i in range(len(self.dataframe.data_dict[self.dependent_variable])):
      value = self.dataframe.data_dict[self.dependent_variable][i]
      prediction = self.predict({k:v[i] for k, v in self.dataframe.data_dict.items() if k != self.dependent_variable})
      rss += (value-prediction)**2
    return rss

  def set_coefficients(self, coefficients):
    self.coefficients = coefficients

  def calc_gradient(self, delta):
    logreg_1 = self.copy()
    logreg_2 = self.copy()
    coeffs = self.coefficients.copy()
    gradient = {}
    for key in coeffs:
      coeffs_1 = coeffs.copy()
      coeffs_2 = coeffs.copy()
      coeffs_1[key] += delta/2
      coeffs_2[key] -= delta/2
      logreg_1.set_coefficients(coeffs_1)
      logreg_2.set_coefficients(coeffs_2)
      gradient[key] = (logreg_1.calc_rss() - logreg_2.calc_rss()) / delta 
    return gradient
  
  def gradient_descent(self, alpha, delta, num_steps, debug_mode=False):
    for i in range(num_steps):
      gradient = self.calc_gradient(delta)
      if debug_mode:
        print("Step "+str(i)+":")
        print("\tGradient: "+str(gradient))
        print("\tCoeffs: "+str(self.coefficients))
        print("\tRSS: "+str(self.calc_rss()))
      for key in gradient:
        self.coefficients[key] -= gradient[key] * alpha


