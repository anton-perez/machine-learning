import sys
sys.path.append('src')
from polynomial_regressor import PolynomialRegressor
from dataframe import DataFrame

data = [(1, 3.1), (2, 10.17), (3, 20.93), (4, 38.71), (5, 60.91), (6, 98.87), (7, 113.92), (8, 146.95), (9, 190.09), (10, 232.65)]

df = DataFrame.from_array(data, ['time', 'distance'])

quadratic_regressor = PolynomialRegressor(degree=2)
quadratic_regressor.fit(df, 'distance')
print('Quadratic Regressor:')
print(quadratic_regressor.coefficients)

for t in [5, 10, 200]:
  print('Distance after '+str(t)+' seconds:',quadratic_regressor.predict({'time':t}))

df = DataFrame.from_array(data, ['time', 'distance'])

cubic_regressor = PolynomialRegressor(degree=3)
cubic_regressor.fit(df, 'distance')
print('Cubic Regressor:')
print(cubic_regressor.coefficients)

for t in [5, 10, 200]:
  print('Distance after '+str(t)+' seconds:',cubic_regressor.predict({'time':t}))