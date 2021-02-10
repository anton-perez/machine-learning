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

# print('Testing attribute coefficients...')
# assert [round(i,5) for i in regressor.coefficients] == [0.01667, 0.15] 
# print('PASSED')

# print('Testing method predict...')
# assert round(regressor.predict({'hours worked': 4}),5) == 0.61667
# print('PASSED')

#Assignment 40
df = DataFrame.from_array(
    [[0, 0, 0.1],
     [1, 0, 0.2],
     [0, 2, 0.5],
     [4,5,0.6]],
    columns = ['scoops of chocolate', 'scoops of vanilla', 'taste rating']
)
regressor = LinearRegressor(df, dependent_variable='taste rating')

print('Testing attribute coefficients...')
assert {key:round(regressor.coefficients[key], 8) for key in regressor.coefficients} == {
    'constant': 0.19252336,
    'scoops of chocolate': -0.05981308,
    'scoops of vanilla': 0.13271028
}, {key:round(regressor.coefficients[key], 7) for key in regressor.coefficients}
print('PASSED')
# these coefficients are rounded, you should only round 
# in your assert statement

print('Testing method predict...')
assert round(regressor.predict({
    'scoops of chocolate': 2,
    'scoops of vanilla': 3
    }),8) == 0.47102804
print('PASSED')


df = DataFrame.from_array(
    [[0, 0, [],               1],
    [0, 0, ['mayo'],          1],
    [0, 0, ['jelly'],         4],
    [0, 0, ['mayo', 'jelly'], 0],
    [5, 0, [],                4],
    [5, 0, ['mayo'],          8],
    [5, 0, ['jelly'],         1],
    [5, 0, ['mayo', 'jelly'], 0],
    [0, 5, [],                5],
    [0, 5, ['mayo'],          0],
    [0, 5, ['jelly'],         9],
    [0, 5, ['mayo', 'jelly'], 0],
    [5, 5, [],                0],
    [5, 5, ['mayo'],          0],
    [5, 5, ['jelly'],         0],
    [5, 5, ['mayo', 'jelly'], 0]],
    columns = ['beef', 'pb', 'condiments', 'rating']
)

df = df.create_dummy_variables('condiments')

df = df.create_interaction_terms('beef','pb')
df = df.create_interaction_terms('beef','mayo')
df = df.create_interaction_terms('beef','jelly')
df = df.create_interaction_terms('pb','mayo')
df = df.create_interaction_terms('pb','jelly')
df = df.create_interaction_terms('mayo', 'jelly')

linear_regressor = LinearRegressor(df, 'rating')
print('Linear Regressor',linear_regressor.coefficients)

print('Testing interaction terms incorporation in predict methods...')
# test 8 slices of beef + mayo
observation = {'beef': 8, 'mayo': 1}
assert round(linear_regressor.predict(observation),2) == 11.34


# test 4 tbsp of pb + 8 slices of beef + mayo
observation = {'beef': 8, 'pb': 4, 'mayo': 1}
assert round(linear_regressor.predict(observation),2) == 3.62


# test 8 slices of beef + mayo + jelly
observation = {'beef': 8, 'mayo': 1, 'jelly': 1}
assert round(linear_regressor.predict(observation),2) == 2.79

print('PASSED')