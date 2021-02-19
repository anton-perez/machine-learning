import sys
sys.path.append('src')
from polynomial_regressor import PolynomialRegressor
from matrix import Matrix
from dataframe import DataFrame

df = DataFrame.from_array(
    [(0,1), (1,2), (2,5), (3,10), (4,20), (5,30)],
    columns = ['x', 'y']
)

print('Testing degree 0 polynomial regressor...')
constant_regressor = PolynomialRegressor(degree=0)
constant_regressor.fit(df, dependent_variable='y')
assert [round(constant_regressor.coefficients[n], 4) for n in constant_regressor.coefficients]== [11.3333]
assert round(constant_regressor.predict({'x': 2}), 4) == 11.3333
print('PASSED')

print('Testing degree 1 polynomial regressor...')
linear_regressor = PolynomialRegressor(degree=1)
linear_regressor.fit(df, dependent_variable='y')
assert [round(linear_regressor.coefficients[n], 4) for n in linear_regressor.coefficients] == [-3.2381, 5.8286]
assert round(linear_regressor.predict({'x': 2}), 4) == 8.4190
print('PASSED')

print('Testing degree 2 polynomial regressor...')
quadratic_regressor = PolynomialRegressor(degree=2)
quadratic_regressor.fit(df, dependent_variable='y')
assert [round(quadratic_regressor.coefficients[n], 4) for n in quadratic_regressor.coefficients] == [1.1071, -0.6893, 1.3036]
assert round(quadratic_regressor.predict({'x': 2}), 4) == 4.9429
print('PASSED')

print('Testing degree 3 polynomial regressor...')
cubic_regressor = PolynomialRegressor(degree=3)
cubic_regressor.fit(df, dependent_variable='y')
assert [round(cubic_regressor.coefficients[n], 4) for n in cubic_regressor.coefficients] == [1.1349, -0.8161, 1.3730, -0.0093]
assert round(cubic_regressor.predict({'x': 2}), 4) == 4.9206
print('PASSED')

print('Testing degree 5 polynomial regressor...')
quintic_regressor = PolynomialRegressor(degree=5)
quintic_regressor.fit(df, dependent_variable='y')
assert [round(quintic_regressor.coefficients[n], 4) for n in quintic_regressor.coefficients] == [1.0000, -2.9500, 6.9583, -3.9583, 1.0417, -0.0917]
assert round(quintic_regressor.predict({'x': 2}), 4) == 5.0000
print('PASSED')
