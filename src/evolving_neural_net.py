import math
from numpy import random
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from neural_network import *


class EvolvingNeuralNets:
  def __init__(self, num_nets, layer_sizes, activation_function, datapoints, mutation_rate):
    self.neural_nets = [NeuralNetwork(self.generate_weight_dict(layer_sizes), activation_function, datapoints, mutation_rate) for _ in range(num_nets)]
    self.act_f = activation_function
    self.datapoints = datapoints
    self.num_nets = num_nets
    self.avg_rss_list = []

  def generate_weight_dict(self, layer_sizes, bias = True):
    weight_dict = {}
    num_nodes = 1
    current_layer = [i+num_nodes for i in range(layer_sizes[0])]
    num_nodes += layer_sizes[0]
    for layer_size in layer_sizes[1:]:
      next_layer = [i+num_nodes for i in range(layer_size)]
      num_nodes += layer_size
      if bias:
        next_layer.append(num_nodes)
        num_nodes += 1
      for idx, parent in enumerate(next_layer):
        for child in current_layer:
          if idx != len(next_layer)-1:
            weight_dict[(parent, child)] = random.normal()
          else:
            if not bias:
              weight_dict[(parent, child)] = random.normal() 
      current_layer = next_layer    
    return weight_dict
    
  def select_top_nets(self, neural_nets):
    rss_list = [net.rss() for net in neural_nets]
    top_idxs = sorted(zip(rss_list, [i for i in range(self.num_nets)]), reverse=False)[:math.floor(self.num_nets/2)]
    top_nets = [neural_nets[i] for rss, i in top_idxs]
    return top_nets

  def generate_generation(self, neural_nets):
    top_nets = self.select_top_nets(neural_nets)
    new_generation = top_nets.copy()
    for neural_net in top_nets:
      new_generation.append(self.produce_child(neural_net))
    return new_generation

  def produce_child(self, neural_net):
    parent_weights = neural_net.weights.copy()
    parent_mut_rate = neural_net.mutation_rate
    num_weights = len(parent_weights)
    child_weights = {edge:weight+parent_mut_rate*random.normal() for edge, weight in parent_weights.items()}
    child_mut_rate = parent_mut_rate*math.exp(random.normal()/((2**0.5)*num_weights**0.25))
    return NeuralNetwork(child_weights, self.act_f, self.datapoints, child_mut_rate)

  def run_generations(self, n):
    for i in range(n):
      print("generation:",i)
      self.neural_nets = self.generate_generation(self.neural_nets)
      self.avg_rss_list.append(self.average_rss(self.neural_nets))
      
  def average_rss(self, neural_nets):
    rss_list = [net.rss() for net in neural_nets]
    return sum(rss_list)/len(rss_list)
    
    
    

