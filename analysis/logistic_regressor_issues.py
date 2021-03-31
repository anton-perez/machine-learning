import sys
sys.path.append('src')
from logistic_regressor import LogisticRegressor
from matrix import *
from dataframe import *
import matplotlib.pyplot as plt

points = [[1,0],
  [2,0],
  [3,0],
  [2,1],
  [3,1],
  [4,1]]

df = DataFrame.from_array(points, ['x', 'y'])

def change_1s_0s_to(x, zero_val, one_val):
  if x == 0:
    return zero_val
  elif x == 1:
    return one_val
  return x

df1 = df.apply('y', (lambda x: change_1s_0s_to(x, 0.1, 0.9)))
regressor1 = LogisticRegressor(df1, 'y', 1)

df2 = df.apply('y', (lambda x: change_1s_0s_to(x, 0.01, 0.99)))
regressor2 = LogisticRegressor(df2, 'y', 1)

df3 = df.apply('y', (lambda x: change_1s_0s_to(x, 0.001, 0.999)))
regressor3 = LogisticRegressor(df3, 'y', 1)

df4 = df.apply('y', (lambda x: change_1s_0s_to(x, 0.0001, 0.9999)))
regressor4 = LogisticRegressor(df4, 'y', 1)

plt.clf()
plt.style.use('bmh')
plt.plot(
  [point[0] for point in points], 
  [point[1] for point in points])

plt.plot(
  [x/1000 for x in range(5001)], 
  [regressor1.predict({'x':x/1000}) for x in range(5001)], label='0.1')

plt.plot(
  [x/1000 for x in range(5001)], 
  [regressor2.predict({'x':x/1000}) for x in range(5001)], label='0.01')

plt.plot(
  [x/1000 for x in range(5001)], 
  [regressor3.predict({'x':x/1000}) for x in range(5001)], label='0.001')

plt.plot(
  [x/1000 for x in range(5001)], 
  [regressor4.predict({'x':x/1000}) for x in range(5001)], label='0.0001')

plt.legend()

plt.savefig('log.png')

df = DataFrame.from_array(
    [[1,0],
    [2,0],
    [3,0],
    [2,1],
    [3,1],
    [4,1]],
    columns = ['x', 'y'])

reg = LogisticRegressor(df, dependent_variable='y')

reg.set_coefficients({'constant': 0.5, 'x': 0.5})

alpha = 0.01
delta = 0.01
num_steps = 20000
reg.gradient_descent(alpha, delta, num_steps)

print(reg.coefficients) 

#{'constant': 2.7911, 'x': -1.1165}

plt.clf()
plt.style.use('bmh')
plt.plot(
  [point[0] for point in points], 
  [point[1] for point in points])

plt.plot(
  [x/1000 for x in range(5001)], 
  [reg.predict({'x':x/1000}) for x in range(5001)])

plt.savefig('optlog.png')
