import sys
sys.path.append('src')
from gradient_descent import GradientDescent

def single_variable_function(x):
  return (x-1)**2
def two_variable_function(x, y):
  return (x-1)**2 + (y-1)**3
def three_variable_function(x, y, z):
  return (x-1)**2 + (y-1)**3 + (z-1)**4
def six_variable_function(x1, x2, x3, x4, x5, x6):
  return (x1-1)**2 + (x2-1)**3 + (x3-1)**4 + x4 + 2*x5 + 3*x6


minimizer = GradientDescent(f=single_variable_function, initial_point=[0])
assert minimizer.point == [0]
print('Testing gradient on single_variable_function...')
assert [round(i,3) for i in minimizer.compute_gradient(delta=0.01)] == [-2.000]
print('PASSED')
minimizer.descend(alpha=0.001, delta=0.01, num_steps=1)
print('Testing descend on single_variable_function...')
assert [round(i,3) for i in minimizer.point] == [0.002]
print('PASSED')

minimizer = GradientDescent(f=two_variable_function, initial_point=[0,0])
assert minimizer.point == [0,0]
print('Testing gradient on two_variable_function...')
assert [round(i,3) for i in minimizer.compute_gradient(delta=0.01)] == [-2.000, 3.000]
print('PASSED')
minimizer.descend(alpha=0.001, delta=0.01, num_steps=1)
print('Testing descend on two_variable_function...')
assert [round(i,3) for i in minimizer.point] == [0.002, -0.003]
print('PASSED')

minimizer = GradientDescent(f=three_variable_function, initial_point=[0,0,0])
assert minimizer.point == [0,0,0]
print('Testing gradient on three_variable_function...')
assert [round(i,3) for i in minimizer.compute_gradient(delta=0.01)] == [-2.000, 3.000, -4.000]
print('PASSED')
minimizer.descend(alpha=0.001, delta=0.01, num_steps=1)
print('Testing descend on three_variable_function...')
assert [round(i,3) for i in minimizer.point] == [0.002, -0.003, 0.004]
print('PASSED')

minimizer = GradientDescent(f=six_variable_function, initial_point=[0,0,0,0,0,0])
assert minimizer.point == [0,0,0,0,0,0]
print('Testing gradient on six_variable_function...')
assert [round(i,3) for i in minimizer.compute_gradient(delta=0.01)] == [-2.000, 3.000, -4.000, 1.000, 2.000, 3.000]
print('PASSED')
minimizer.descend(alpha=0.001, delta=0.01, num_steps=1)
print('Testing descend on six_variable_function...')
assert [round(i,3) for i in minimizer.point] == [0.002, -0.003, 0.004, -0.001, -0.002, -0.003]
print('PASSED')