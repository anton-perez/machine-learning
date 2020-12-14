import sys
sys.path.append('src')
from matrix import Matrix
from dataframe import DataFrame

class LinearRegressor:
  def __init__(self, dataframe, dependent_variable):
    self.dataframe = dataframe
    self.dependent_variable = dependent_variable
    self.independent_variables = [var for var in self.dataframe.columns if var != self.dependent_variable]
    self.coefficients = self.calculate_coefficients()
    
  def calculate_coefficients(self):
    data = self.dataframe.data_dict
    indep_coeff = [data[self.independent_variables[i]] for i in range(len(self.independent_variables))]
    dependent = Matrix([data[self.dependent_variable]]).transpose()
    coeff_mat = Matrix([[1] for _ in range(len(indep_coeff[0]))]).augment(Matrix(indep_coeff).transpose())
    psuedo = coeff_mat.transpose()@coeff_mat
    final_coeff = psuedo.inverse()@(coeff_mat.transpose()@dependent) 
    dict_keys = ['constant'] + self.independent_variables

    return {dict_keys[i]:final_coeff.transpose().elements[0][i] for i in range(len(dict_keys))}
  
  def predict(self, input_dict):
    sum = self.coefficients['constant']
    for var in self.independent_variables:
      sum += self.coefficients[var]*input_dict[var]
    return sum