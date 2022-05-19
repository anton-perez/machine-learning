import math
import matplotlib.pyplot as plt

datapoints = {(0,5),(2,3),(5,10)}

weights = {(1,3):1,
           (1,4):1,
           (2,3):1,
           (2,4):1,
           (3,6):1,
           (4,6):1,
           (5,6):1}

def act_f(x):
  return max(0,x)

def predict(w, x):
  return act_f(w[(3,6)]*act_f(w[(1,3)]*act_f(x)+w[(2,3)])+w[(4,6)]*act_f(w[(1,4)]*act_f(x)+w[(2,4)])+w[(5,6)])


inputs = [i/100 for i in range(800)]
outputs = [predict(weights, i/100) for i in range(800)]

plt.clf()
plt.style.use('bmh')
plt.title("Neural Network Regression")
plt.scatter([0,2,5],[5,3,10], label='Datapoints')
plt.plot(inputs, outputs, label="Initial Regression")

def rss(w):
  rss = 0
  for point in datapoints:
    value = point[1]
    prediction = predict(w, point[0])
    rss += (value-prediction)**2
  return rss

def compute_gradient(f, weights, delta):
  gradient = {}
  for edge in weights:
    partial = 0
    temp_point = weights.copy()
    temp_point[edge] += delta/2
    partial += f(temp_point)
    temp_point[edge] -= delta
    partial -= f(temp_point)
    gradient[edge] = partial/delta

  return gradient

RSS_vals = []

def descend(f, initial_weights, alpha, delta, num_steps):
  for step in range(num_steps):
    for edge in initial_weights:
      initial_weights[edge] -= alpha*compute_gradient(f, initial_weights, delta)[edge]
    if step%100 == 0:
      print('RSS at step ', step, ': ', rss(initial_weights))
      RSS_vals.append(rss(initial_weights))
      print('Parameters: ', initial_weights)


descend(rss, weights, 0.01, 0.00001, 2000)
print(weights)

outputs = [predict(weights, i/100) for i in range(800)]

plt.plot(inputs, outputs, label="Final Regression")
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.legend()

plt.savefig('neural_net_practice_output.png')

plt.clf()
plt.title("Neural Network RSS")
plt.plot([100*i for i in range(20)], RSS_vals)
plt.xlabel('Number of Iterations')
plt.ylabel('RSS')
plt.savefig('neural_net_practice_RSS.png')


