import math
import matplotlib.pyplot as plt

class Node:
  def __init__(self, index):
    self.index = index
    self.children = []
    self.parents = []
    self.input = None
    self.output = None
    self.bias = False
    self.previous = None
    self.distance = None

class NeuralNetwork:
  def __init__(self, weights, activation_function, datapoints):
    self.weights = weights
    self.edges = [edge for edge in self.weights]
    self.f = activation_function
    self.datapoints = datapoints
    self.max_index = self.get_max_index()
    self.nodes = [Node(i+1) for i in range(self.max_index)]
    self.build_from_edges()

  def get_children(self, index):
    children = []
    for edge in self.edges:
      if edge[0] == index:
        children.append(edge[1])
    return children

  def get_parents(self, index):
    parents = []
    for edge in self.edges:
      if edge[1] == index:
        parents.append(edge[0])
    return parents

  def get_max_index(self):
    max = self.edges[0][0]
    for edge in self.edges:
      if edge[0] > max:
        max = edge[0]
      elif edge[1] > max:
        max = edge[1]
    return max

  def get_node_from_index(self, index):
    return self.nodes[index-1]

  def build_from_edges(self):
    for node in self.nodes:
      children = [self.get_node_from_index(i) for i in self.get_children(node.index)]
      parents = [self.get_node_from_index(i) for i in self.get_parents(node.index)]
      node.children = children
      node.parents = parents
      for child in children:
        if node not in child.parents:
          child.parents.append(node)
      
      for parent in parents:
        if node not in parent.children:
          parent.children.append(node)

      if children == [] and node.index != 1:
        node.bias = True
        node.output = 1

  def nodes_breadth_first(self, index):
    queue = [self.nodes[index]]
    visited = []
    while queue != []:
      visiting = queue[0]
      queue = queue[1:]
      visited.append(visiting)
      children = visiting.children
      queue = queue + [child 
                       for child in children 
                       if child not in queue and child not in visited]
    
    return visited

  def clear_network(self):
    for node in self.nodes:
      if not node.bias:
        node.input = None
        node.output = None

  def foward_propagate(self, input):
    self.clear_network()
    input_node = self.get_node_from_index(1)
    input_node.input = input
    queue = [input_node]
    visited = []
    while queue != []:
      visiting = queue[0]
      if visiting.input == None and visiting.bias == False:
        visiting.input = 0
        end_index = visiting.index
        for child in visiting.children:
          start_index = child.index
          visiting.input += self.weights[(end_index, start_index)]*child.output
      visiting.output = self.f(visiting.input)

      queue = queue[1:]
      visited.append(visiting)
      parents = visiting.parents
      queue = queue + [parent 
                       for parent in parents 
                       if parent not in queue and parent not in visited]
    
    return self.nodes[-1].output

  def rss(self):
    rss = 0
    for point in self.datapoints:
      value = point[1]
      prediction = self.foward_propagate(point[0])
      rss += (prediction-value)**2
    return rss

  def f_prime(self, input):
    delta = 0.000000001
    return (self.f(input+delta/2) - self.f(input-delta/2))/delta

  def calculate_neuron_gradient(self, datapoint):
    gradient = {i+1:0 for i in range(len(self.nodes))}
    self.foward_propagate(datapoint[0])
    for node in self.nodes[::-1]:
      if node.index == len(self.nodes):
        gradient[node.index] = 2*(node.output-datapoint[1])
      else:
        for parent in node.parents:
          gradient[node.index] += gradient[parent.index]*self.f_prime(parent.input)*self.weights[(parent.index,node.index)]
      
    return gradient

  def calculate_weight_gradient(self):
    gradient = {edge:0 for edge in self.edges}
    for datapoint in self.datapoints:
      temp_gradient = {edge:0 for edge in self.edges}
      neuron_grad = self.calculate_neuron_gradient(datapoint)
      for edge in self.weights:
        end, start = edge
        start_node = self.get_node_from_index(start)
        end_node = self.get_node_from_index(end)
        temp_gradient[edge] = neuron_grad[end]*self.f_prime(end_node.input)*start_node.output
      for edge in temp_gradient:
        gradient[edge] += temp_gradient[edge]
    
    return gradient

  def backward_propagate(self, alpha, num_steps):
    for step in range(num_steps):
      weight_gradient = self.calculate_weight_gradient()
      if step%100 == 0:
      #   print('RSS at step ', step, ': ', self.rss())
      #   print('Parameters: ', self.weights)
      #   print('Gradient: ', weight_gradient)
        RSS_vals.append(self.rss())
      
      for edge in self.weights:
        self.weights[edge] -= alpha*weight_gradient[edge]
      


datapoints = [(-5,-3),(-4,-1),(-3,1),(-2,2),(-1,-1),(1,-1),(2,1),(3,2),(4,3),(5,4),(6,2),(7,0)]

weights = {(3,1):0,
           (4,1):0,
           (3,2):0,
           (4,2):0,
           (6,3):0,
           (6,4):0,
           (6,5):0,
           (7,3):0,
           (7,4):0,
           (7,5):0,
           (9,6):0,
           (9,7):0,
           (9,8):0}

for edge in weights:
  k, j = edge
  weights[edge] = ((-1)**(j+k))*min(j,k)/max(j,k) 

act_f = lambda x: max(0,x)

neural_net = NeuralNetwork(weights, act_f, datapoints)
print(neural_net.calculate_weight_gradient())
# print(neural_net.weights)
# neural_net.backward_propagate(0.001, 1)
# print(neural_net.weights)

inputs = [i/100 - 10 for i in range(2000)]
outputs = [neural_net.foward_propagate(i/100 - 10) for i in range(2000)]


plt.clf()
plt.style.use('bmh')
plt.title("Neural Network Regression")
plt.scatter([point[0] for point in datapoints],[point[1] for point in datapoints], label='Datapoints')
plt.plot(inputs, outputs, label="Initial Regression")


RSS_vals = []


neural_net.backward_propagate(0.001, 2000)
print(neural_net.weights)

outputs = [neural_net.foward_propagate(i/100 - 10) for i in range(2000)]

plt.plot(inputs, outputs, label="Final Regression")
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.legend()

plt.savefig('neural_net_analysis_output.png')

plt.clf()
plt.title("Neural Network RSS")
plt.plot([100*i for i in range(20)], RSS_vals)
plt.xlabel('Number of Iterations')
plt.ylabel('RSS')
plt.savefig('neural_net_analysis_RSS.png')