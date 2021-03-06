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
    if indep_coeff != []:
      coeff_mat = Matrix([[1] for _ in range(len(data[self.dependent_variable]))]).augment(Matrix(indep_coeff).transpose())
    else:
      coeff_mat = Matrix([[1] for _ in range(len(data[self.dependent_variable]))])
    psuedo = coeff_mat.transpose()@coeff_mat
    final_coeff = psuedo.inverse()@(coeff_mat.transpose()@dependent) 
    dict_keys = ['constant'] + self.independent_variables

    return {dict_keys[i]:final_coeff.transpose().elements[0][i] for i in range(len(dict_keys))}
  
  def predict(self, input_dict):
    sum = self.coefficients['constant']
    for var in self.independent_variables:
      if var in input_dict:
        sum += self.coefficients[var]*input_dict[var]
      elif ' * ' in var:
        vars = var.split(' * ')
        if vars[0] in input_dict and vars[1] in input_dict:
          sum += self.coefficients[var]*input_dict[vars[0]]*input_dict[vars[1]]
    return sum