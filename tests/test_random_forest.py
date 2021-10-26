import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from random_forest import *



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


def get_folds(df, fold_num):
  df = df.sample(frac=1).reset_index(drop=True)
  return [df.iloc[int(i*len(df.index)/fold_num):int((i+1)*len(df.index)/fold_num)] for i in range(fold_num)]

def calc_accuracy(df, tree_num, fold_num):
  print('Tree Num',tree_num)
  correct_classifications = 0
  total_classifications = len(df.index)
  folds = get_folds(df, fold_num)
  for i in range(fold_num): 
    test_df = folds[i].copy().reset_index(drop=True)
    train_df = pd.concat([folds[j] for j in range(fold_num) if j != i])
    print('fold ',i)
    rf = RandomForest(tree_num, 10)
    rf.fit(train_df)

    for j in range(len(test_df.index)):
      vars = [var for var in test_df.columns if var != 'class']
      datapoint = pd.DataFrame(test_df[vars].iloc[j]).T.reset_index(drop=True)
      print(datapoint)
      print(datapoint.columns)

      predicted_cls = rf.predict(datapoint)
      actual_cls = test_df.iloc[j]['class']

      if predicted_cls == actual_cls:
        correct_classifications += 1
  
  return correct_classifications/total_classifications

tree_nums = [1,10,20,50,100,500]
accuracies = [calc_accuracy(df, i, 5) for i in tree_nums]

print(accuracies)
plt.clf()
plt.style.use('bmh')
plt.plot(tree_nums, accuracies)
plt.xlabel('tree_nums')
plt.ylabel('Accuracy')
plt.xticks(tree_nums)

plt.savefig('k_fold_random_forest.png')