import sys
sys.path.append('src')
from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
from dataframe import DataFrame

df = DataFrame.from_array(
  [['Shortbread'  ,     0.14     ,       0.14     ,      0.28     ,     0.44      ],
  ['Shortbread'  ,     0.10     ,       0.18     ,      0.28     ,     0.44      ],
  ['Shortbread'  ,     0.12     ,       0.10     ,      0.33     ,     0.45      ],
  ['Shortbread'  ,     0.10     ,       0.25     ,      0.25     ,     0.40      ],
  ['Sugar'       ,     0.00     ,       0.10     ,      0.40     ,     0.50      ],
  ['Sugar'       ,     0.00     ,       0.20     ,      0.40     ,     0.40      ],
  ['Sugar'       ,     0.10     ,       0.08     ,      0.35     ,     0.47      ],
  ['Sugar'       ,     0.00     ,       0.05     ,      0.30     ,     0.65      ],
  ['Fortune'     ,     0.20     ,       0.00     ,      0.40     ,     0.40      ],
  ['Fortune'     ,     0.25     ,       0.10     ,      0.30     ,     0.35      ],
  ['Fortune'     ,     0.22     ,       0.15     ,      0.50     ,     0.13      ],
  ['Fortune'     ,     0.15     ,       0.20     ,      0.35     ,     0.30      ],
  ['Fortune'     ,     0.22     ,       0.00     ,      0.40     ,     0.38      ]],
  columns = ['Cookie Type' ,'Portion Eggs','Portion Butter','Portion Sugar','Portion Flour' ]
  )

knn = KNearestNeighborsClassifier(k=5)
knn.fit(df, dependent_variable = 'Cookie Type')
observation = {
  'Portion Eggs': 0.10,
  'Portion Butter': 0.15,
  'Portion Sugar': 0.30,
  'Portion Flour': 0.45
}

print(knn.compute_distances(observation).to_array())
# Returns a dataframe representation of the following array:

# [[0.047, 'Shortbread'],
#  [0.037, 'Shortbread'],
#  [0.062, 'Shortbread'],
#  [0.122, 'Shortbread'],
#  [0.158, 'Sugar'],
#  [0.158, 'Sugar'],
#  [0.088, 'Sugar'],
#  [0.245, 'Sugar'],
#  [0.212, 'Fortune'],
#  [0.187, 'Fortune'],
#  [0.396, 'Fortune'],
#  [0.173, 'Fortune'],
#  [0.228, 'Fortune']]

# Note: the above has been rounded to 3 decimal places for ease of viewing, but you should not round in your
# actual class.

print(knn.nearest_neighbors(observation).to_array())
# Returns a dataframe representation of the following array:

# [[0.037, 'Shortbread'],
#  [0.047, 'Shortbread'],
#  [0.062, 'Shortbread'],
#  [0.088, 'Sugar'],
#  [0.122, 'Shortbread'],
#  [0.158, 'Sugar'],
#  [0.158, 'Sugar'],
#  [0.173, 'Fortune'],
#  [0.187, 'Fortune'],
#  [0.212, 'Fortune'],
#  [0.228, 'Fortune'],
#  [0.245, 'Sugar'],
#  [0.396, 'Fortune']]

print(knn.classify(observation))
#'Shortbread' # because this is the majority class
             # in the 5 nearest neighbors

df = DataFrame.from_array(
  [['A', 0],
  ['A', 1],
  ['B', 2],
  ['B', 3]],
  columns = ['letter', 'number']
)

knn = KNearestNeighborsClassifier(k=4)
knn.fit(df, dependent_variable = 'letter')
observation = {
  'number': 1.6
}
print(knn.classify(observation))
#'B'

df = DataFrame.from_array(
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


def get_df_without_row(df, row_index):
  rows = [i for i in range(len(df.to_array())) if i != row_index]
  return df.select_rows(rows)

def leave_one_out_cross_validation_accuracy(df, dependent_variable, k): 
  correct_classfications = 0
  total_classifications = len(df.to_array())
  for i in range(total_classifications):
    knn = KNearestNeighborsClassifier(k)
    knn.fit(get_df_without_row(df, i), dependent_variable)
    left_out = {k:v[i] for k,v in df.data_dict.items() if k != dependent_variable} 
    predicted_classification = knn.classify(left_out)
    actual_classification = df.data_dict[dependent_variable][i]

    if predicted_classification == actual_classification:
      correct_classfications += 1

  return correct_classfications/total_classifications

accuracies = [round(leave_one_out_cross_validation_accuracy(df, 'Cookie Type', k), 2) for k in range(1,19)]

print(accuracies)

    




# def get_accuracy():
