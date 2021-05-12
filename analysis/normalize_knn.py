import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/eurisko-us/eurisko-us.github.io/master/files/datasets/book-data.csv')

simple_df = df.copy()
minmax_df = df.copy()
z_score_df = df.copy()

features = [col for col in df.columns if col != 'book type']
for feature in features:
  simple_df[feature] = simple_df[feature]/simple_df[feature].max()
  minmax_df[feature] = (minmax_df[feature]-minmax_df[feature].min()) / (minmax_df[feature].max()-minmax_df[feature].min())
  z_score_df[feature] = (z_score_df[feature]-z_score_df[feature].mean()) / z_score_df[feature].std()

def leave_one_out_cross_validation_accuracy(df, dependent_variable, k): 
  correct_classfications = 0
  total_classifications = len(df.to_numpy().tolist())
  for i in range(total_classifications):
    independent_df = df[[col for col in df.columns if col != dependent_variable]]
    dependent_df = df[dependent_variable]

    left_out = independent_df.iloc[[i]].to_numpy().tolist()[0]
    actual_classification = dependent_df.iloc[[i]].to_numpy().tolist()[0]

    independent = independent_df.drop([i]).reset_index(drop=True).to_numpy().tolist()
    dependent = dependent_df.drop([i]).reset_index(drop=True).to_numpy().tolist()
    
    knn = knearestclass(n_neighbors=k)
    knn = knn.fit(independent, dependent)
    predicted_classification = knn.predict([left_out])

    if predicted_classification == actual_classification:
      correct_classfications += 1

  return correct_classfications/total_classifications

k_vals = [2*n+1 for n in range(50)]
unnormalized_accuracies = [round(leave_one_out_cross_validation_accuracy(df, 'book type', k), 2) for k in k_vals]
simple_accuracies = [round(leave_one_out_cross_validation_accuracy(simple_df, 'book type', k), 2) for k in k_vals]
minmax_accuracies = [round(leave_one_out_cross_validation_accuracy(minmax_df, 'book type', k), 2) for k in k_vals]
z_score_accuracies = [round(leave_one_out_cross_validation_accuracy(z_score_df, 'book type', k), 2) for k in k_vals]


plt.style.use('bmh')
plt.plot(k_vals, unnormalized_accuracies, label="unnormalized")
plt.plot(k_vals, simple_accuracies, label="simple scaling")
plt.plot(k_vals, minmax_accuracies, label="min-max")
plt.plot(k_vals, z_score_accuracies, label="z-scoring")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Leave One Out Cross Accuracy for Various Normalization')
plt.legend()
plt.savefig('normalized_leave_one_out_accuracy.png')