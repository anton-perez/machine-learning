class SimplexAlgorithm:
  def __init__(self, array, start_vars):
    self.array = array
    self.start_vars = start_vars
    self.num_rows = len(self.array)
    self.num_cols = len(self.array[0])
    self.max = 0
    self.objective_vals = []

  def copy(self):
    return [[i for i in row] for row in self.array]
  
  def normalize_row(self, row_index, pivot_index):
    normalized_copy = self.copy()
    # first_index = normalized_copy.nonzero_in_row(row_index)
    pivot = normalized_copy[row_index][pivot_index]
    for col_index in range(self.num_cols):
      normalized_copy[row_index][col_index] *= 1/pivot
    return normalized_copy
  
  def clear_below(self, row_index, pivot_index):
    cleared_copy = self.copy()
    pivot_val = cleared_copy[row_index][pivot_index]
    for below_index in range(row_index + 1, self.num_rows):
      scalar = cleared_copy[below_index][pivot_index]/pivot_val
      for col_index in range(self.num_cols):
        cleared_copy[below_index][col_index] -= scalar*cleared_copy[row_index][col_index]
    return cleared_copy

  def clear_above(self, row_index, pivot_index):
    cleared_copy = self.copy()
    pivot_val = cleared_copy[row_index][pivot_index]
    for above_index in range(row_index):
      scalar = cleared_copy[above_index][pivot_index]/pivot_val
      for col_index in range(self.num_cols):
        cleared_copy[above_index][col_index] -= scalar*cleared_copy[row_index][col_index]
    return cleared_copy

  def determine_strictest_row_index(self, array, col_index):
    min_constraint = 100000000000
    min_index = 0
    for i, row in enumerate(array[:-1]):
      constraint = row[-1]/row[col_index]
      if constraint < min_constraint and constraint > 0:
        min_constraint = constraint
        min_index = i
    return min_index

  def determine_max_partial_col_index(self, array):
    max_partial = array[-1][0]
    max_index = 0
    for i, partial in enumerate(array[-1]):
      if partial > max_partial:
        max_partial = partial
        max_index = i
    return max_index

  def find_max(self):
    while max(self.array[-1]) > 0:
      pivot_index = self.determine_max_partial_col_index(self.array)
      row_index = self.determine_strictest_row_index(self.array, pivot_index)
      self.array = self.normalize_row(row_index, pivot_index)
      self.array = self.clear_below(row_index, pivot_index)
      self.array = self.clear_above(row_index, pivot_index)
    self.max = -self.array[-1][-1]

  def solutions(self):
    self.find_max()
    solutions = {'objective value':self.max}
    for i in range(self.start_vars):
      for row in self.array[:-1]:
        if row[i] == 1:
          solutions[f'x{i+1}'] = row[-1]
          break
    return solutions
    