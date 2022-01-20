from sklearn.tree import DecisionTreeClassifier
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


def get_folds(df, fold_num):
  df = df.sample(frac=1).reset_index(drop=True)
  return [df.iloc[int(i*len(df.index)/fold_num):int((i+1)*len(df.index)/fold_num)] for i in range(fold_num)]

def calc_accuracy(df, min_size_to_split, fold_num):
  correct_classifications = 0
  total_classifications = len(df.index)
  folds = get_folds(df, fold_num)
  vars = [col for col in df.columns if col != 'class']
  for i in range(fold_num): 
    test_df = folds[i].copy().reset_index(drop=True)
    train_df = pd.concat([folds[j] for j in range(fold_num) if j != i])
    print('fold ',i)

    X_train = train_df[vars]
    y_train = train_df['class']

    X_test = test_df[vars]
    y_test = test_df['class']

    dt = DecisionTreeClassifier(min_samples_split= min_size_to_split)
    dt.fit(X_train, y_train)

    predictions = dt.predict(X_test)

    for j in range(len(y_test.index)):

      predicted_cls = predictions[j]
      actual_cls = y_test.iloc[j]

      if predicted_cls == actual_cls:
        correct_classifications += 1
  
  return correct_classifications/total_classifications

min_sizes = [2,5,10,15,20,30,50,100]
accuracies = [calc_accuracy(df, i, 5) for i in min_sizes]

print(accuracies)
plt.clf()
plt.style.use('bmh')
plt.plot(min_sizes, accuracies)
plt.xlabel('min_sizes')
plt.ylabel('Accuracy')
plt.xticks(min_sizes)

plt.savefig('sklearn_k_fold_decision_tree.png')