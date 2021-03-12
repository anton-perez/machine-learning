class DataFrame:
  def __init__(self, data_dict, column_order):
    self.data_dict = data_dict
    self.columns = column_order

  def to_array(self): 
    column_length = len(self.data_dict[self.columns[0]])
    data_array = [[0 for key in self.columns] for row in range(column_length)]

    col_index = 0
    for key in self.columns:
      for index in range(len(self.data_dict[key])):
        data_array[index][col_index] = self.data_dict[key][index]
      col_index += 1

    return data_array

  def select_columns(self, selected_columns):
    new_dict = {key:self.data_dict[key] for key in selected_columns}
    return DataFrame(new_dict, selected_columns)

  def select_rows(self, row_indices):
    new_dict = self.data_dict.copy()
    for key in new_dict:
      new_dict[key] = [self.data_dict[key][i] for i in row_indices]

    return DataFrame(new_dict, self.columns)
  
  def apply(self, key, function):
    new_dict = self.data_dict.copy()
    old_list = new_dict[key]
    new_dict[key] = [function(i) for i in old_list]
    return DataFrame(new_dict, self.columns)

  @classmethod
  def from_array(cls, arr, columns):
    data_dict = {columns[i]:[arr[j][i] for j in range(len(arr))] for i in range(len(columns))}
    return cls(data_dict, columns)

  def array_row_to_dict(self, row):
    return {self.columns[i]:row[i] for i in range(len(row))}

  def select_rows_where(self, condition):
    arr = self.to_array()
    selected_rows = []
    for row in arr:
      row_dict = self.array_row_to_dict(row)
      if condition(row_dict):
        selected_rows.append(row)
    
    return DataFrame.from_array(selected_rows, self.columns)

  def order_by(self, key, ascending):
    arr = self.to_array()
    new_arr = []
    for iterations in range(len(arr)):
      min_row = arr[0]
      min = self.array_row_to_dict(min_row)[key]
      for row in arr:
        row_dict = self.array_row_to_dict(row)
        if row_dict[key] < min:
          min = row_dict[key]
          min_row = row
      new_arr.append(min_row)
      arr.remove(min_row)

    if not ascending:
      new_arr = new_arr[::-1]

    return DataFrame.from_array(new_arr, self.columns)

  @classmethod
  def from_csv(cls, path_to_csv, data_types, parser, header=True):
    with open(path_to_csv, "r") as file:
      str_arr = [i.split(',  ') for i in file.read().split('\n')]
      columns = parser(str_arr[0][0].split(', ')[0])
      parsed_str_arr = [parser(row[0]) for row in str_arr[1:] if row != ['']]
      data_str_arr = [[data_types[columns[i]](row[i]) if row[i] != '' else None for i in range(len(columns))] for row in parsed_str_arr]
      
    return cls.from_array(data_str_arr, columns)

  def create_interaction_terms(self, col_1, col_2):
    new_dict = self.data_dict.copy()
    new_columns = [i for i in self.columns]
    new_key = col_1 + ' * ' + col_2
    new_columns.append(new_key)
    new_dict[new_key] = [new_dict[col_1][i]*new_dict[col_2][i] for i in range(len(new_dict[col_1]))]

    return DataFrame(new_dict, new_columns)
  
  def create_dummy_variables(self, column):
    new_dict = self.data_dict.copy()
    dummy_col = new_dict[column]
    new_dict.pop(column)
    dummy_vars = []
    for var_list in dummy_col:
      for var in var_list:
        if var not in new_dict:
          new_dict[var] = []
          dummy_vars.append(var)

    for dummy_var in dummy_vars:
      for var_list in dummy_col:
        if dummy_var in var_list:
          new_dict[dummy_var].append(1)
        else:
          new_dict[dummy_var].append(0)

    col_index = self.columns.index(column)    
    new_columns = self.columns[:col_index] + dummy_vars + self.columns[col_index + 1:]

    return DataFrame(new_dict, new_columns)