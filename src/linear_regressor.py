import sys
sys.path.append('src')
from matrix import Matrix
from dataframe import DataFrame

class LinearRegressor:
  def __init__(self, dataframe, dependent_variable):
    self.dataframe = dataframe
    self.dependent_variable = dependent_variable
    self.independent_variable = self.dataframe.columns[0] if self.dataframe.columns[0] != self.dependent_variable else self.dataframe.columns[1]
    self.coefficients = self.calculate_coefficients()

  def calculate_coefficients(self):
    data = self.dataframe.data_dict
    indep_coeff = data[self.independent_variable]
    dependent = Matrix([data[self.dependent_variable]]).transpose()
    coeff_mat = Matrix([[1] for _ in range(len(indep_coeff))]).augment(Matrix([indep_coeff]).transpose())
    psuedo = coeff_mat.transpose()@coeff_mat
    final_coeff = psuedo.inverse()@(coeff_mat.transpose()@dependent) 

    return final_coeff.transpose().elements[0]
  
  def predict(self, input):
    return self.coefficients[0] + self.coefficients[1]*input[self.independent_variable]