import sys
sys.path.append('src')
from dataframe import *
from linear_regressor import *
from polynomial_regressor import *

dataset = [(-4, 11.0),
 (-2, 5.0),
 (0, 3.0),
 (2, 5.0),
 (4, 11.1),
 (6, 21.1),
 (8, 35.1),
 (10, 52.8),
 (12, 74.8),
 (14, 101.2)]

training_data = [dataset[i] for i in range(len(dataset)) if i % 2 == 0]
testing_data = [dataset[i] for i in range(len(dataset)) if i % 2 != 0]

training_set_df = DataFrame.from_array(training_data, ['x', 'y'])
testing_set_df = DataFrame.from_array(testing_data, ['x', 'y'])

linear_regressor = LinearRegressor(training_set_df, 'y')
print('Linear Regressor coefficients:',linear_regressor.coefficients)

quadratic_regressor = PolynomialRegressor(degree=2)
quadratic_regressor.fit(training_set_df, 'y')
print('Quadratic Regressor coefficients:',quadratic_regressor.coefficients)

cubic_regressor = PolynomialRegressor(degree=3)
cubic_regressor.fit(training_set_df, 'y')
print('Cubic Regressor coefficients:',cubic_regressor.coefficients)

quartic_regressor = PolynomialRegressor(degree=4)
quartic_regressor.fit(training_set_df, 'y')
print('Quartic Regressor coefficients:',quartic_regressor.coefficients)

linear_train_rss = 0
for point in training_data:
  linear_train_rss += (point[1] - linear_regressor.predict({'x':point[0]}))**2
print('Linear Regressor training data RSS:', linear_train_rss)

linear_test_rss = 0
for point in testing_data:
  linear_test_rss += (point[1] - linear_regressor.predict({'x':point[0]}))**2
print('Linear Regressor testing data RSS:', linear_test_rss)

quadratic_train_rss = 0
for point in training_data:
  quadratic_train_rss += (point[1] - quadratic_regressor.predict({'x':point[0]}))**2
print('Quadratic Regressor training data RSS:', quadratic_train_rss)

quadratic_test_rss = 0
for point in testing_data:
  quadratic_test_rss += (point[1] - quadratic_regressor.predict({'x':point[0]}))**2
print('Quadratic Regressor testing data RSS:', quadratic_test_rss)

cubic_train_rss = 0
for point in training_data:
  cubic_train_rss += (point[1] - cubic_regressor.predict({'x':point[0]}))**2
print('Cubic Regressor training data RSS:', cubic_train_rss)

cubic_test_rss = 0
for point in testing_data:
  cubic_test_rss += (point[1] - cubic_regressor.predict({'x':point[0]}))**2
print('Cubic Regressor testing data RSS:', cubic_test_rss)

quartic_train_rss = 0
for point in training_data:
  quartic_train_rss += (point[1] - quartic_regressor.predict({'x':point[0]}))**2
print('Quartic Regressor training data RSS:', quartic_train_rss)

quartic_test_rss = 0
for point in testing_data:
  quartic_test_rss += (point[1] - quartic_regressor.predict({'x':point[0]}))**2
print('Quartic Regressor testing data RSS:', quartic_test_rss)