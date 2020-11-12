import sys
sys.path.append('src')
from dataframe import DataFrame

data_dict = {
    'Pete': [1, 0, 1, 0],
    'John': [2, 1, 0, 2],
    'Sarah': [3, 1, 4, 0]
}

df1 = DataFrame(data_dict, column_order = ['Pete', 'John', 'Sarah'])
print('Testing attribute "data_dict"...')
assert df1.data_dict == {
    'Pete': [1, 0, 1, 0],
    'John': [2, 1, 0, 2],
    'Sarah': [3, 1, 4, 0]
}
print('PASSED')

print('Testing attribute "columns"...')
assert df1.columns == ['Pete', 'John', 'Sarah']
print('PASSED')

print('Testing method "to_array"...')
assert df1.to_array() == [[1, 2, 3],
 [0, 1, 1],
 [1, 0, 4],
 [0, 2, 0]]
print('PASSED')

df2 = df1.select_columns(['Sarah', 'Pete'])

print('Testing method "select_columns"...')
assert df2.to_array() == [[3, 1],
 [1, 0],
 [4, 1],
 [0, 0]]
assert df2.columns == ['Sarah', 'Pete']
print('PASSED')

df3 = df1.select_rows([1,3])
print('Testing method "select_rows"...')
assert df3.to_array() == [[0, 1, 1],
                   [0, 2, 0]]
print('PASSED')

data_dict = {
    'Pete': [1, 0, 1, 0],
    'John': [2, 1, 0, 2],
    'Sarah': [3, 1, 4, 0]
}

df1 = DataFrame(data_dict, column_order = ['Pete', 'John', 'Sarah'])
df2 = df1.apply('John', lambda x: 7 * x)
print('Testing method "apply"...')
assert df2.data_dict == {
    'Pete': [1, 0, 1, 0],
    'John': [14, 7, 0, 14],
    'Sarah': [3, 1, 4, 0]
}
print('PASSED')

columns = ['firstname', 'lastname', 'age']
arr = [['Kevin', 'Fray', 5],
           ['Charles', 'Trapp', 17],
           ['Anna', 'Smith', 13],
           ['Sylvia', 'Mendez', 9]]
df = DataFrame.from_array(arr, columns)

print('Testing method "select_rows_where"...')
assert df.select_rows_where(
    lambda row: len(row['firstname']) >= len(row['lastname']) and row['age'] > 10
    ).to_array() == [['Charles', 'Trapp', 17]]
print('PASSED')

print('Testing method "order_by"...')
assert df.order_by('age', ascending=True).to_array() ==[['Kevin', 'Fray', 5],
['Sylvia', 'Mendez', 9],
['Anna', 'Smith', 13],
['Charles', 'Trapp', 17]]

assert df.order_by('firstname', ascending=False).to_array() == [['Sylvia', 'Mendez', 9],
['Kevin', 'Fray', 5],
['Charles', 'Trapp', 17],
['Anna', 'Smith', 13]]
print('PASSED')