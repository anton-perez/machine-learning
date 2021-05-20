import sys
sys.path.append('src')
from k_means import *
import matplotlib.pyplot as plt

columns = ['Portion Eggs',
            'Portion Butter',
            'Portion Sugar',
            'Portion Flour']

data = [[0.14, 0.14, 0.28, 0.44],
        [0.22, 0.1, 0.45, 0.33],
        [0.1, 0.19, 0.25, 0.4],
        [0.02, 0.08, 0.43, 0.45],
        [0.16, 0.08, 0.35, 0.3],
        [0.14, 0.17, 0.31, 0.38],
        [0.05, 0.14, 0.35, 0.5],
        [0.1, 0.21, 0.28, 0.44],
        [0.04, 0.08, 0.35, 0.47],
        [0.11, 0.13, 0.28, 0.45],
        [0.0, 0.07, 0.34, 0.65],
        [0.2, 0.05, 0.4, 0.37],
        [0.12, 0.15, 0.33, 0.45],
        [0.25, 0.1, 0.3, 0.35],
        [0.0, 0.1, 0.4, 0.5],
        [0.15, 0.2, 0.3, 0.37],
        [0.0, 0.13, 0.4, 0.49],
        [0.22, 0.07, 0.4, 0.38],
        [0.2, 0.18, 0.3, 0.4]]

def calc_rss(point1, point2):
  return sum([(point1[i] - point2[i])**2 for i in range(len(point1))])

def calc_k_means_error(k_val, data):
  initial_clusters = {k+1:[i for i in range(k, len(data), k_val)] for k in range(k_val)}
  kmeans = KMeans(initial_clusters, data)
  kmeans.run()
  total_squared_error = 0
  for center in kmeans.centers:
    center_squared_error = 0
    for row_index in kmeans.clusters[center]:
      center_squared_error += calc_rss(kmeans.centers[center], data[row_index])
    total_squared_error += center_squared_error

  return total_squared_error

k_vals = [k for k in range(1,6)]
error = [calc_k_means_error(k, data) for k in k_vals]

plt.style.use('bmh')
plt.plot(k_vals, error)
plt.xlabel('k')
plt.ylabel('sum squared error')
plt.xticks(k_vals)
plt.savefig('elbow_method.png')
