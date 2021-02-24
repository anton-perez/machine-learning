import sys
sys.path.append('src')
from logistic_regressor import LogisticRegressor
from dataframe import DataFrame

data = [[10, 0.05], [100, 0.35], [1000, 0.95]]

df = DataFrame.from_array(data, ['x','y'])

regressor = LogisticRegressor(df, 'y', 1)

print(regressor.coefficients)
print(regressor.predict({
    'x': 500
    }))