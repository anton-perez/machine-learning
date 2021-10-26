import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from random_decision_tree import *

print('Test 1')
df = pd.DataFrame(data = [[1,1,1],[1,2,1],[2,1,1],[2,2,2],[2,3,2],[3,1,2],[3,2,2]], columns = ['x','y','class'])

dt = RandomDecisionTree(1)
dt.fit(df)

print('Root:')
print(dt.root.data)
print('First Split:')
print(dt.root.split_var, '=',dt.root.split_val)
print(dt.root.children[0].data)
print(dt.root.children[1].data)

print('Test 2')
datapoints = []
def add_cluster_datapoints(n, r_max, center, classification, data_list):
  for i in range(n):
    x,y = center
    r = r_max*random.random()**2
    theta = random.uniform(0, 2*math.pi)
    new_point = [x+r*math.cos(theta), y+r*math.sin(theta), classification]
    data_list.append(new_point)
     
add_cluster_datapoints(50, 3, (1,1), 'X', datapoints)
add_cluster_datapoints(50, 3, (4,4), 'X', datapoints)
add_cluster_datapoints(50, 3, (1,4), 'O', datapoints)
add_cluster_datapoints(50, 3, (4,1), 'O', datapoints)

X_x_list = [datapoints[i][0] for i in range(100)]
X_y_list = [datapoints[i][1] for i in range(100)]
O_x_list = [datapoints[i][0] for i in range(100,200)]
O_y_list = [datapoints[i][1] for i in range(100,200)]

# plt.style.use('bmh')
# plt.plot(X_x_list, X_y_list, 'rx')
# plt.plot(O_x_list, O_y_list, 'bo')
# plt.savefig('scattered_data.png')

df = pd.DataFrame(data = datapoints, columns = ['x','y','class'])
dt = RandomDecisionTree(1)
dt.fit(df)
