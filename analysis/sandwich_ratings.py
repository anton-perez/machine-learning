import sys
sys.path.append('src')
from linear_regressor import LinearRegressor
from matrix import Matrix
from dataframe import DataFrame

df = DataFrame.from_array(
    [[0, 0, 1], 
    [1, 0, 2], 
    [2, 0, 4], 
    [4, 0, 8], 
    [6, 0, 9], 
    [0, 2, 2], 
    [0, 4, 5], 
    [0, 6, 7], 
    [0, 8, 6]],
    columns = ['slices of roast beef', 'tablespoons of peanut butter', 'rating']
)
regressor = LinearRegressor(df, dependent_variable='rating')

print(regressor.coefficients)
print(regressor.predict({
    'slices of roast beef': 5,
    'tablespoons of peanut butter': 0
    }))

print(regressor.predict({
    'slices of roast beef': 5,
    'tablespoons of peanut butter': 5
    }))

df = DataFrame.from_array(
    [[0, 0, 1], 
    [1, 0, 2], 
    [2, 0, 4], 
    [4, 0, 8], 
    [6, 0, 9], 
    [0, 2, 2], 
    [0, 4, 5], 
    [0, 6, 7], 
    [0, 8, 6],
    [2, 2, 0],
    [3, 4, 0]],
    columns = ['beef', 'pb', 'rating']
)

df = df.create_interaction_terms('beef', 'pb')

regressor = LinearRegressor(df, dependent_variable='rating')

print('regressor with interaction terms')

print(regressor.coefficients)
print(regressor.predict({
    'beef': 5,
    'pb': 0,
    'beef * pb': 0
    }))

print(regressor.predict({
    'beef': 5,
    'pb': 5,
    'beef * pb': 25
    }))
