import sys
sys.path.append('src')
from linear_regressor import LinearRegressor
from logistic_regressor import LogisticRegressor
from matrix import Matrix
from dataframe import DataFrame


df = DataFrame.from_array(
    [[1,0.2],
     [2,0.25],
     [3,0.5]],
    columns = ['x','y']
)

log_reg = LogisticRegressor(df, dependent_variable = 'y')
print('Testing method predict...')
assert round(log_reg.predict({'x': 5}), 3) == 0.777
print('PASSED')