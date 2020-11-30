import math


class GradientDescent:
  def __init__(self, f, initial_point):
    self.f = f
    self.initial_point = initial_point
    self.point = initial_point
    self.arg_num = f.__code__.co_argcount

  def compute_gradient(self, delta):
    gradient = []
    for i in range(self.arg_num):
      partial = 0
      temp_point = [i for i in self.point]
      temp_point[i] += delta/2
      partial += self.f(*temp_point)
      temp_point[i] -= delta
      partial -= self.f(*temp_point)
      gradient.append(partial/delta)
    #print(gradient)
    return gradient
  
  def descend(self, alpha, delta, num_steps):
    for step in range(num_steps):
      for i in range(self.arg_num):
        self.point[i] -= alpha*self.compute_gradient(delta)[i]
      #print(self.point)


