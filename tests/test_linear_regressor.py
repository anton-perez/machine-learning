import sys
sys.path.append('src')
from linear_regressor import LinearRegressor
from matrix import Matrix
from dataframe import DataFrame

df = DataFrame.from_array(
    [[1,0.2],
     [2,0.25],
     [3,0.5]],
    columns = ['hours worked', 'progress']
)
regressor = LinearRegressor(df, dependent_variable='progress')

print('Testing attribute coefficients...')
assert [round(i,5) for i in regressor.coefficients] == [0.01667, 0.15] 
print('PASSED')

print('Testing method predict...')
assert round(regressor.predict({'hours worked': 4}),5) == 0.61667
print('PASSED')