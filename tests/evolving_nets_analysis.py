import math
from numpy import random
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from neural_network import *
from evolving_neural_net import *

datapoints = [(0.0, 7), (0.2, 5.6), (0.4, 3.56), (0.6, 1.23), (0.8, -1.03), (1.0, -2.89), (1.2, -4.06), (1.4, -4.39), (1.6, -3.88), (1.8, -2.64), (2.0, -0.92), (2.2, 0.95), (2.4, 2.63), (2.6, 3.79), (2.8, 4.22), (3.0, 3.8), (3.2, 2.56), (3.4, 0.68), (3.6, -1.58), (3.8, -3.84), (4.0, -5.76), (4.2, -7.01), (4.4, -7.38), (4.6, -6.76), (4.8, -5.22)]

def normalize_list(list, interval):
  a,b = interval
  max_val = max(list)
  min_val = min(list)
  normal_list = [((b-a)*(i-min_val)/(max_val-min_val))+a for i in list]
  return normal_list

unzip_data = list(zip(*datapoints))
normal_unzip_data = [normalize_list(unzip_data[0], [0,1]), normalize_list(unzip_data[1], [-1,1])]
normal_data = list(zip(normal_unzip_data[0], normal_unzip_data[1]))

print(normal_data)

def act_f(x):
  return math.tanh(x)

enn = EvolvingNeuralNets(30, [1,10,6,3,1], act_f, normal_data, 0.5)
num_gens = 3000

inputs = [i/100 for i in range(120)]

plt.clf()
plt.style.use('bmh')
plt.title("Evolving Neural Networks Regressions")
plt.scatter(normal_unzip_data[0],normal_unzip_data[1], label='Datapoints')
#initial regression
for neural_net in enn.neural_nets:
  outputs = [neural_net.foward_propagate(i/100) for i in range(120)]
  plt.plot(inputs, outputs, color="red")

enn.run_generations(num_gens)

#final regression
for neural_net in enn.neural_nets:
  outputs = [neural_net.foward_propagate(i/100) for i in range(120)]
  plt.plot(inputs, outputs, color="blue")
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.legend()

plt.savefig('evolving_nets_output.png')

plt.clf()
plt.title("Evolving Neural Networks RSS")
plt.plot([i for i in range(num_gens)], enn.avg_rss_list)
plt.xlabel('Number of Iterations')
plt.ylabel('RSS')
plt.savefig('evolving_nets_RSS.png')
