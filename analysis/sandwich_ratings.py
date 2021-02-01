import sys
sys.path.append('src')
from linear_regressor import LinearRegressor
from logistic_regressor import LogisticRegressor
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
    [2, 2, 0.1],
    [3, 4, 0.1]],
    columns = ['beef', 'pb', 'rating']
)

df = df.create_interaction_terms('beef', 'pb')

regressor = LogisticRegressor(df, 'rating', 10)

print('Logistic regressor with interaction terms')

print(regressor.coefficients)
print(regressor.predict({
    'beef': 5,
    'pb': 0,
    'beef * pb': 0
    }))

print(regressor.predict({
    'beef': 12,
    'pb': 0,
    'beef * pb': 0
    }))

print(regressor.predict({
    'beef': 5,
    'pb': 5,
    'beef * pb': 25
    }))

print('-----------Assignment 52-----------')

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

df = DataFrame.from_array(
    [[0, 0, [],               1],
    [0, 0, ['mayo'],          1],
    [0, 0, ['jelly'],         4],
    [0, 0, ['mayo', 'jelly'], 0.1],
    [5, 0, [],                4],
    [5, 0, ['mayo'],          8],
    [5, 0, ['jelly'],         1],
    [5, 0, ['mayo', 'jelly'], 0.1],
    [0, 5, [],                5],
    [0, 5, ['mayo'],          0.1],
    [0, 5, ['jelly'],         9],
    [0, 5, ['mayo', 'jelly'], 0.1],
    [5, 5, [],                0.1],
    [5, 5, ['mayo'],          0.1],
    [5, 5, ['jelly'],         0.1],
    [5, 5, ['mayo', 'jelly'], 0.1]],
    columns = ['beef', 'pb', 'condiments', 'rating']
)

df = df.create_dummy_variables('condiments')

df = df.create_interaction_terms('beef','pb')
df = df.create_interaction_terms('beef','mayo')
df = df.create_interaction_terms('beef','jelly')
df = df.create_interaction_terms('pb','mayo')
df = df.create_interaction_terms('pb','jelly')
df = df.create_interaction_terms('mayo', 'jelly')


logistic_regressor = LogisticRegressor(df, 'rating', 10) 
print('Logistic Regressor',logistic_regressor.coefficients)

print('8 slices of beef + mayo')
print('Linear', linear_regressor.predict({
    'beef': 8,
    'pb': 0,
    'mayo': 1,
    'jelly': 0,
    'beef * pb': 0,
    'beef * mayo': 8,
    'beef * jelly': 0,
    'pb * mayo': 0,
    'pb * jelly': 0,
    'mayo * jelly': 0
    }))
print('Logistic', logistic_regressor.predict({
    'beef': 8,
    'pb': 0,
    'mayo': 1,
    'jelly': 0,
    'beef * pb': 0,
    'beef * mayo': 8,
    'beef * jelly': 0,
    'pb * mayo': 0,
    'pb * jelly': 0,
    'mayo * jelly': 0
    }))

print('4 tbsp of pb + jelly')
print('Linear', linear_regressor.predict({
    'beef': 0,
    'pb': 4,
    'mayo': 0,
    'jelly': 1,
    'beef * pb': 0,
    'beef * mayo': 0,
    'beef * jelly': 0,
    'pb * mayo': 0,
    'pb * jelly': 4,
    'mayo * jelly': 0
    }))
print('Logistic', logistic_regressor.predict({
    'beef': 0,
    'pb': 4,
    'mayo': 0,
    'jelly': 1,
    'beef * pb': 0,
    'beef * mayo': 0,
    'beef * jelly': 0,
    'pb * mayo': 0,
    'pb * jelly': 4,
    'mayo * jelly': 0
    }))

print('4 tbsp of pb + mayo')
print('Linear', linear_regressor.predict({
    'beef': 0,
    'pb': 4,
    'mayo': 1,
    'jelly': 0,
    'beef * pb': 0,
    'beef * mayo': 0,
    'beef * jelly': 0,
    'pb * mayo': 4,
    'pb * jelly': 0,
    'mayo * jelly': 0
    }))
print('Logistic', logistic_regressor.predict({
    'beef': 0,
    'pb': 4,
    'mayo': 1,
    'jelly': 0,
    'beef * pb': 0,
    'beef * mayo': 0,
    'beef * jelly': 0,
    'pb * mayo': 4,
    'pb * jelly': 0,
    'mayo * jelly': 0
    }))

print('4 tbsp of pb + 8 slices of beef + mayo')
print('Linear', linear_regressor.predict({
    'beef': 8,
    'pb': 4,
    'mayo': 1,
    'jelly': 0,
    'beef * pb': 32,
    'beef * mayo': 8,
    'beef * jelly': 0,
    'pb * mayo': 4,
    'pb * jelly': 0,
    'mayo * jelly': 0
    }))
print('Logistic', logistic_regressor.predict({
    'beef': 8,
    'pb': 4,
    'mayo': 1,
    'jelly': 0,
    'beef * pb': 32,
    'beef * mayo': 8,
    'beef * jelly': 0,
    'pb * mayo': 4,
    'pb * jelly': 0,
    'mayo * jelly': 0
    }))


print('8 slices of beef + mayo + jelly')
print('Linear', linear_regressor.predict({
    'beef': 8,
    'pb': 0,
    'mayo': 1,
    'jelly': 1,
    'beef * pb': 0,
    'beef * mayo': 8,
    'beef * jelly': 8,
    'pb * mayo': 0,
    'pb * jelly': 0,
    'mayo * jelly': 1
    }))
print('Logistic', logistic_regressor.predict({
    'beef': 8,
    'pb': 0,
    'mayo': 1,
    'jelly': 1,
    'beef * pb': 0,
    'beef * mayo': 8,
    'beef * jelly': 8,
    'pb * mayo': 0,
    'pb * jelly': 0,
    'mayo * jelly': 1
    }))
