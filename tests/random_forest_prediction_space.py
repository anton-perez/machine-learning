from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from random_subset_forest import *



print('Test 1')
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

plt.style.use('bmh')
plt.plot(X_x_list, X_y_list, 'rx')
plt.plot(O_x_list, O_y_list, 'bo')
plt.savefig('scattered_data.png')

df = pd.DataFrame(data = datapoints, columns = ['x','y','class'])

rf = RandomForestClassifier(n_estimators=100)

vars = [var for var in df.columns if var != 'class']

X = df[vars]
y = df['class']

rf.fit(X,y)

all_points = [[x/10, y/10] for x in range(-20,70) for y in range(-20,70)]
all_points_df = pd.DataFrame(data = all_points, columns = ['x', 'y'])

predictions = rf.predict(all_points_df)
X_prediction_points = [all_points[i] for i in range(len(all_points)) if predictions[i]=='X']
O_prediction_points = [all_points[i] for i in range(len(all_points)) if predictions[i]=='O']

X_prediction_x = [p[0] for p in X_prediction_points]
X_prediction_y = [p[1] for p in X_prediction_points]
O_prediction_x = [p[0] for p in O_prediction_points]
O_prediction_y = [p[1] for p in O_prediction_points]

plt.scatter(X_prediction_x,X_prediction_y,c = 'r', s=1.5, marker = '.')
plt.scatter(O_prediction_x,O_prediction_y,c = 'b', s=1.5,marker = '.')

plt.savefig('random_forest_prediction_space.png')