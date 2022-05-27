import sys
sys.path.append('src')
from simplex_algorithm import *

array1 = [[2,1,1,1,0,0,14],
          [4,2,3,0,1,0,28],
          [2,5,5,0,0,1,30],
          [1,2,1,0,0,0,0]]


array2 = [[3,2,5,1,0,0,0,55],
          [2,1,1,0,1,0,0,26],
          [1,1,3,0,0,1,0,30],
          [5,2,4,0,0,0,1,57],
          [20,10,15,0,0,0,0,0]]

sa = SimplexAlgorithm(array1, 3)

print(sa.solutions())

sa = SimplexAlgorithm(array2, 3)

print(sa.solutions())