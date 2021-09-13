import pandas as pd
import sys
sys.path.append('src')
from decision_tree import *

df = pd.DataFrame(data = [[1,1,1],[1,2,1],[2,1,1],[2,2,2],[2,3,2],[3,1,2],[3,2,2]], columns = ['x','y','class'])

dt = DecisionTree()
dt.fit(df)
print(dt.data)
print(dt.split_vals)

point = pd.DataFrame(data = [[0.5,0.5]], columns = ['x','y'])
print(dt.predict(point))
point = pd.DataFrame(data = [[2.5,2.5]], columns = ['x','y'])
print(dt.predict(point))
point = pd.DataFrame(data = [[1.6,1.6]], columns = ['x','y'])
print(dt.predict(point))
