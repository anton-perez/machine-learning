import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from decision_tree import *

print('Test 1')
df = pd.DataFrame(data = [[1,1,1],[1,2,1],[2,1,1],[2,2,2],[2,3,2],[3,1,2],[3,2,2]], columns = ['x','y','class'])

dt = DecisionTree(1)
dt.fit(df)
# print(dt.data)
# print(dt.split_vals)
print('Root:')
print(dt.root.data)
print('First Split:')
print(dt.root.split_var, '=',dt.root.split_val)
print(dt.root.children[0].data)
print(dt.root.children[1].data)

print('Second Split:')
print(dt.root.children[1].split_var, '=',dt.root.children[1].split_val)
print(dt.root.children[1].children[0].data)
print(dt.root.children[1].children[1].data)

print('Testing decision tree predict method...')
point = pd.DataFrame(data = [[0.5,0.5]], columns = ['x','y'])
assert dt.predict(point) == 1 #should be 1
point = pd.DataFrame(data = [[2.5,2.5]], columns = ['x','y'])
assert dt.predict(point) == 2 #should be 2
point = pd.DataFrame(data = [[1.6,1.6]], columns = ['x','y'])
assert dt.predict(point) == 2 #should be 2
print('PASSED')


print('Test 2')
df = pd.DataFrame(data = [
  [1,9,'O'],
  [1,7,'X'],
  [2,7,'X'],
  [3,7,'X'],
  [3,8,'X'],
  [3,9,'X'],
  [5,1,'O'],
  [5,2,'O'],
  [5,3,'O'],
  [6,3,'O'],
  [7,3,'O'],
  [7,1,'X']], columns = ['x','y','class'])

dt = DecisionTree(7)
dt.fit(df)

print('Root:')
print(dt.root.data)
print('First Split:')
print(dt.root.split_var, '=',dt.root.split_val)
print(dt.root.children[0].data)
print(dt.root.children[1].data)


print('Testing min_size_to_split...')
assert dt.root.children[0].children == []
assert dt.root.children[1].children == []
point = pd.DataFrame(data = [[2,8]], columns = ['x','y'])
assert dt.predict(point) == 'X' #should be X
point = pd.DataFrame(data = [[6,2]], columns = ['x','y'])
assert dt.predict(point) == 'O' #should be O
print('PASSED')


print('Test 3')
df = pd.DataFrame(data = [
  [1,1,'X'],
  [2,1,'O'],
  [1,2,'O'],
  [2,2,'X'],
  [1,3,'X'],
  [2,3,'X']], columns = ['x','y','class'])

dt = DecisionTree(5)
dt.fit(df)

print('Root:')
print(dt.root.data)
print('First Split:')
print(dt.root.split_var, '=',dt.root.split_val)
print(dt.root.children[0].data)
print(dt.root.children[1].data)

print('Testing min_size_to_split...')
assert dt.root.children[0].children == []
assert dt.root.children[1].children == []
point = pd.DataFrame(data = [[3,1]], columns = ['x','y'])
assert dt.predict(point) == 'O' #should be O
point = pd.DataFrame(data = [[3,4]], columns = ['x','y'])
assert dt.predict(point) == 'X' #should be X
print('PASSED')


print('Test 4')
df = pd.DataFrame(data = [
  [0,1,'X'],
  [0,1,'X'],
  [0,2,'X'],
  [0,2,'O'],
  [1,1,'X'],
  [1,1,'O'],
  [1,1,'O'],
  [1,2,'X'],
  [1,2,'X'],
  [1,2,'O']], columns = ['x','y','class'])

dt = DecisionTree(1)
dt.fit(df)

print('Root:')
print(dt.root.data)
print('First Split:')
print(dt.root.split_var, '=',dt.root.split_val)
print(dt.root.children[0].data)
print(dt.root.children[1].data)

print('Testing multiple datapoints on one point...')
point = pd.DataFrame(data = [[3,5]], columns = ['x','y'])
assert dt.predict(point) == 'X' #should be X
point = pd.DataFrame(data = [[0,0]], columns = ['x','y'])
assert dt.predict(point) == 'X' #should be X
point = pd.DataFrame(data = [[3,1]], columns = ['x','y'])
assert dt.predict(point) == 'O' #should be O
print('PASSED')

print('Test 5')
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

def calc_accuracy(df, min_size_to_split, fold_num):
  correct_classifications = 0
  total_classifications = len(df.index)
  folds = get_folds(df, fold_num)
  for i in range(fold_num): 
    test_df = folds[i].copy().reset_index(drop=True)
    train_df = pd.concat([folds[j] for j in range(fold_num) if j != i])
    print('fold ',i)
    dt = DecisionTree(min_size_to_split)
    dt.fit(train_df)

    for j in range(len(test_df.index)):
      vars = [var for var in test_df.columns if var != 'class']
      datapoint = pd.DataFrame(test_df[vars].iloc[j]).T.reset_index(drop=True)
      print(datapoint)
      print(datapoint.columns)

      predicted_cls = dt.predict(datapoint)
      actual_cls = test_df.iloc[j]['class']

      if predicted_cls == actual_cls:
        correct_classifications += 1
  
  return correct_classifications/total_classifications

min_sizes = [1,2,5,10,15,20,30,50,100]
accuracies = [calc_accuracy(df, i, 5) for i in min_sizes]

print(accuracies)
plt.clf()
plt.style.use('bmh')
plt.plot(min_sizes, accuracies)
plt.xlabel('min_sizes')
plt.ylabel('Accuracy')
plt.xticks(min_sizes)

plt.savefig('k_fold_decision_tree.png')
#[0.83, 0.82, 0.87, 0.825, 0.865, 0.82, 0.84, 0.83, 0.56]

      