import sys
sys.path.append('src')
from dataframe import *
from linear_regressor import *
import math
import matplotlib.pyplot as plt

points = [(0.0, 7.0),
 (0.2, 5.6),
 (0.4, 3.56),
 (0.6, 1.23),
 (0.8, -1.03),
 (1.0, -2.89),
 (1.2, -4.06),
 (1.4, -4.39),
 (1.6, -3.88),
 (1.8, -2.64),
 (2.0, -0.92),
 (2.2, 0.95),
 (2.4, 2.63),
 (2.6, 3.79),
 (2.8, 4.22),
 (3.0, 3.8),
 (3.2, 2.56),
 (3.4, 0.68),
 (3.6, -1.58),
 (3.8, -3.84),
 (4.0, -5.76),
 (4.2, -7.01),
 (4.4, -7.38),
 (4.6, -6.76),
 (4.8, -5.22)]

data = [[math.sin(x), math.cos(x), math.sin(2*x), math.cos(2*x), y] for (x,y) in points]
columns = ['sin(x)','cos(x)','sin(2x)','cos(2x)','y']

df = DataFrame.from_array(data, columns)

signal_regressor = LinearRegressor(df, 'y')
print(signal_regressor.coefficients)

plt.clf()
plt.style.use('bmh')
plt.plot(
  [point[0] for point in points], 
  [point[1] for point in points])

plt.plot(
  [x/1000 for x in range(5001)], 
  [signal_regressor.predict({'sin(x)': math.sin(x/1000), 'cos(x)': math.cos(x/1000), 'sin(2x)': math.sin(2*x/1000), 'cos(2x)': math.cos(2*x/1000)}) for x in range(5001)])

plt.savefig('signal.png')