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

  def select(self, selected_columns):
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

  def where(self, condition):
    arr = self.to_array()
    selected_rows = []
    for row in arr:
      row_dict = self.array_row_to_dict(row)
      if condition(row_dict):
        selected_rows.append(row)
    
    return DataFrame.from_array(selected_rows, self.columns)

  def order_by(self, key, order = 'ASC'):
    ascending = True if order == 'ASC' else False
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

  def group_by(self, column):
    already_seen = []
    new_arr = []
    for i in range(len(self.data_dict[column])):
      old_row = self.select_rows([i]).to_array()[0]
      if self.data_dict[column][i] in already_seen:
        for row in new_arr:
          if self.data_dict[column][i] in row:
            for j in range(len(row)):
              if row[j] != self.data_dict[column][i]:
                row[j].append(old_row[j])
      else:
        already_seen.append(self.data_dict[column][i])
        new_row = [[old_row[j]] if j != self.columns.index(column) else old_row[j] for j in range(len(old_row))]
        new_arr.append(new_row)

    return DataFrame.from_array(new_arr, self.columns)

  def group_by_interval(self, column, intervals):
    new_arr = [[[] if i != self.columns.index(column) else interval for i in range(len(self.columns))] for interval in intervals]
    for i in range(len(self.data_dict[column])):
      if self.data_dict[column][i] != None:
        old_row = self.select_rows([i]).to_array()[0]
        for (a, b) in intervals:
          if self.data_dict[column][i] >= a and self.data_dict[column][i] < b:
            interval = (a,b)

        for row in new_arr:
          if interval in row:
            for j in range(len(row)):
              if j != self.columns.index(column):
                row[j].append(old_row[j])

    return DataFrame.from_array(new_arr, self.columns)

  def aggregate(self, column, how):
    new_data = self.data_dict.copy()
    if how == 'count':
      new_data[column] = [len(group) for group in new_data[column]] 
    if how == 'max':
      new_data[column] = [max(group) for group in new_data[column]]
    if how == 'min':
      new_data[column] = [min(group) for group in new_data[column]]
    if how == 'sum':
      new_data[column] = [sum(group) for group in new_data[column]]
    if how == 'avg':
      new_data[column] = [sum(group)/len(group) for group in new_data[column]]

    return DataFrame(new_data, self.columns)
      
  def query(self, query_string):
    query_string = query_string.replace('ORDER BY', 'ORDERBY')
    operations = ['SELECT', 'ORDERBY']
    query_list = query_string.split(' ')
 
    query_lines = []
    query_line = []
    for word in query_list:
      if word in operations:
        if query_line != []:
          query_lines.append(query_line)
        query_line = [word]
      else:
        query_line.append(word)  
    query_lines.append(query_line)

    query_lines = query_lines[::-1]
    df = DataFrame(self.data_dict, self.columns)
    for line in query_lines:
      if 'SELECT' in line:
        new_columns = [string.replace(',','') for string in line[line.index('SELECT')+1:]]
        df = df.select(new_columns)
      elif 'ORDERBY' in line:
        orders = [string.replace(',','') for string in line[line.index('ORDERBY')+1:]][::-1]
        for order_num in range(0, len(orders), 2):
          df = df.order_by(orders[order_num+1], orders[order_num])
    return df

