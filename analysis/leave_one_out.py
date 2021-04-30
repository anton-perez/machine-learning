import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt

df = pd.DataFrame(
    [['Shortbread'  ,     0.14     ,       0.14     ,      0.28     ,     0.44      ],
['Shortbread'  ,     0.10     ,       0.18     ,      0.28     ,     0.44      ],
['Shortbread'  ,     0.12     ,       0.10     ,      0.33     ,     0.45      ],
['Shortbread'  ,     0.10     ,       0.25     ,      0.25     ,     0.40      ],
['Sugar'       ,     0.00     ,       0.10     ,      0.40     ,     0.50      ],
['Sugar'       ,     0.00     ,       0.20     ,      0.40     ,     0.40      ],
['Sugar'       ,     0.02     ,       0.08     ,      0.45     ,     0.45      ],
['Sugar'       ,     0.10     ,       0.15     ,      0.35     ,     0.40      ],
['Sugar'       ,     0.10     ,       0.08     ,      0.35     ,     0.47      ],
['Sugar'       ,     0.00     ,       0.05     ,      0.30     ,     0.65      ],
['Fortune'     ,     0.20     ,       0.00     ,      0.40     ,     0.40      ],
['Fortune'     ,     0.25     ,       0.10     ,      0.30     ,     0.35      ],
['Fortune'     ,     0.22     ,       0.15     ,      0.50     ,     0.13      ],
['Fortune'     ,     0.15     ,       0.20     ,      0.35     ,     0.30      ],
['Fortune'     ,     0.22     ,       0.00     ,      0.40     ,     0.38      ],
['Shortbread'  ,     0.05     ,       0.12     ,      0.28     ,     0.55      ],
['Shortbread'  ,     0.14     ,       0.27     ,      0.31     ,     0.28      ],
['Shortbread'  ,     0.15     ,       0.23     ,      0.30     ,     0.32      ],
['Shortbread'  ,     0.20     ,       0.10     ,      0.30     ,     0.40      ]],
    columns = ['Cookie Type' ,'Portion Eggs','Portion Butter','Portion Sugar','Portion Flour' ]
    )


def leave_one_out_cross_validation_accuracy(df, dependent_variable, k): 
  correct_classfications = 0
  total_classifications = len(df.to_numpy().tolist())
  for i in range(total_classifications):
    independent_df = df[[col for col in df.columns if col != dependent_variable]]
    dependent_df = df[dependent_variable]

    left_out = independent_df.iloc[[i]].to_numpy().tolist()[0]
    actual_classification = df[dependent_variable].iloc[[i]].to_numpy().tolist()[0]

    independent_df = independent_df.drop([i])
    independent = independent_df.reset_index(drop=True).to_numpy().tolist()
    dependent_df = dependent_df.drop([i])
    dependent = dependent_df.reset_index(drop=True).to_numpy().tolist()
    
    knn = knearestclass(n_neighbors=k)
    knn = knn.fit(independent, dependent)
    predicted_classification = knn.predict([left_out])

    if predicted_classification == actual_classification:
      correct_classfications += 1

  return correct_classfications/total_classifications

k_vals = [k for k in range(1,19)]
accuracies = [round(leave_one_out_cross_validation_accuracy(df, 'Cookie Type', k), 2) for k in range(1,19)]

print(accuracies)

plt.style.use('bmh')
plt.plot(k_vals, accuracies)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.xticks(k_vals)
plt.title('Leave One Out Cross Validation')
plt.savefig('leave_one_out_accuracy.png')
