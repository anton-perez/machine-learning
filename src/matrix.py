class Matrix:
  
  def __init__(self, elements):
    self.elements = elements
    self.num_rows = len(elements)
    self.num_cols = len(elements[0])

  def copy(self):
    copied_elements = [[entry for entry in row] for row in self.elements]  
    return Matrix(copied_elements)

  def add(self, other_matrix):
    other_elements = other_matrix.elements
    sum_elements = [[None for element in range(self.num_cols)] for row in range(self.num_rows)]
    
    for i in range(len(sum_elements)): #rows
      for j in range(len(sum_elements[i])): #cols
        sum_elements[i][j] = self.elements[i][j] + other_elements[i][j]
    
    return Matrix(sum_elements)

  def subtract(self, other_matrix):
    other_elements = other_matrix.elements
    dif_elements = [[None for element in range(self.num_cols)] for row in range(self.num_rows)]

    for i in range(len(dif_elements)):
      for j in range(len(dif_elements[i])):
        dif_elements[i][j] = self.elements[i][j] - other_elements[i][j]
    
    return Matrix(dif_elements)
  
  def scalar_multiply(self, scalar):
    new_elements = [[None for element in range(self.num_cols)] for row in range(self.num_rows)]

    for i in range(len(new_elements)):
      for j in range(len(new_elements[i])):
        new_elements[i][j] = round(scalar*self.elements[i][j], 4)

    return Matrix(new_elements)

  def matrix_multiply(self, other_matrix):
    other_elements = other_matrix.elements
    product_elements = [[0 for element in range(other_matrix.num_cols)] for row in self.elements]

    for i in range(len(product_elements)):
      for j in range(len(product_elements[i])):
        for k in range(len(self.elements[0])):
          product_elements[i][j] += self.elements [i][k]*other_elements[k][j]
    
    return Matrix(product_elements)
  
  def transpose(self):
    transposed_elements = [[0 for element in range(self.num_rows)] for row in range(self.num_cols)]

    for i in range(self.num_rows):
      for j in range(self.num_cols):
        transposed_elements[j][i] = self.elements[i][j]
    
    return Matrix(transposed_elements)
  
  def is_equal(self, other_matrix):
    return self.elements == other_matrix.elements
  
  def get_pivot_row(self, column_index):
    for row_index in range(self.num_rows):
      left_zeros = True
      for left_index in range(column_index):
        if self.elements[row_index][left_index] != 0:
          left_zeros = False
      
      if self.elements[row_index][column_index] != 0 and left_zeros:
        return row_index
    return None

  def swap_rows(self, row_index1, row_index2): 
    swap_copy = self.copy()
    swap_copy.elements[row_index1], swap_copy.elements[row_index2] = swap_copy.elements[row_index2], swap_copy.elements[row_index1]

    return swap_copy

  def nonzero_in_row(self, row_index):
    for col_index in range(self.num_cols):
      if self.elements[row_index][col_index] != 0:
        return col_index

  def normalize_row(self, row_index):
    normalized_copy = self.copy()
    first_index = normalized_copy.nonzero_in_row(row_index)
    first_ele = normalized_copy.elements[row_index][first_index]
    for col_index in range(normalized_copy.num_cols):
      normalized_copy.elements[row_index][col_index] *= 1/first_ele
    return normalized_copy
  
  def clear_below(self, row_index):
    cleared_copy = self.copy()
    j = cleared_copy.nonzero_in_row(row_index)
    j_val = cleared_copy.elements[row_index][j]
    for below_index in range(row_index + 1, cleared_copy.num_rows):
      scalar = cleared_copy.elements[below_index][j]/j_val
      for col_index in range(cleared_copy.num_cols):
        cleared_copy.elements[below_index][col_index] -= scalar*cleared_copy.elements[row_index][col_index]
    return cleared_copy

  def clear_above(self, row_index):
    cleared_copy = self.copy()
    j = cleared_copy.nonzero_in_row(row_index)
    j_val = cleared_copy.elements[row_index][j]
    for above_index in range(row_index):
      scalar = cleared_copy.elements[above_index][j]/j_val
      for col_index in range(cleared_copy.num_cols):
        cleared_copy.elements[above_index][col_index] -= scalar*cleared_copy.elements[row_index][col_index]
    return cleared_copy
  
  def rref(self):
    rref_matrix = self.copy()
    row_index = 0
    for col_index in range(rref_matrix.num_cols):
      pivot_row = rref_matrix.get_pivot_row(col_index)
      if pivot_row != None:
        if pivot_row != row_index:
          rref_matrix = rref_matrix.swap_rows(row_index, pivot_row)
 
        rref_matrix = rref_matrix.normalize_row(row_index)
        rref_matrix = rref_matrix.clear_above(row_index)
        rref_matrix = rref_matrix.clear_below(row_index)

        row_index += 1

    return rref_matrix

  def augment(self, other_matrix):
    other_elements = other_matrix.elements
    augmented_elements = [self.elements[row_index] + other_elements[row_index] for row_index in range(self.num_rows)]
    return Matrix(augmented_elements)

  def get_rows(self, row_nums):
    fetched_rows = [self.elements[row_index] for row_index in row_nums]
    return Matrix(fetched_rows)
  
  def get_columns(self, col_nums):
    fetched_cols = [[self.elements[row_index][col_index] for col_index in col_nums] for row_index in range(self.num_rows)]
    return Matrix(fetched_cols)
  
  def inverse(self):
    identity_matrix = Matrix([[1 if row == col else 0 for col in range(self.num_cols)] for row in range(self.num_rows)])
    augmented_matrix = self.augment(identity_matrix)
    reduced_augmented_matrix = augmented_matrix.rref()
    inverted_matrix = reduced_augmented_matrix.get_columns([i+self.num_cols for i in range(self.num_cols)])
    reduced_matrix = reduced_augmented_matrix.get_columns([i for i in range(self.num_cols)])
    
    rounded_elements = [[round(element,10) for element in row] for row in reduced_matrix.elements]
    if self.num_rows != self.num_cols:
      return "Error: cannot invert a non-square matrix"
    elif rounded_elements != identity_matrix.elements:
      return "Error: cannot invert a singular matrix"
    else:
      return inverted_matrix  

    #return "Error: cannot invert a non-square matrix" if self.num_rows != self.num_cols else "Error: cannot invert a singular matrix" if self.augment(Matrix([[1 if row == col else 0 for col in range(self.num_cols)] for row in range(self.num_rows)])).rref().get_columns([i for i in range(self.num_cols)]).elements !=  [[1 if row == col else 0 for col in range(self.num_cols)] for row in range(self.num_rows)] else self.augment(Matrix([[1 if row == col else 0 for col in range(self.num_cols)] for row in range(self.num_rows)])).rref().get_columns([i+self.num_cols for i in range(self.num_cols)]) 

  def determinant(self):
    det_matrix = self.copy()
    if det_matrix.num_rows == det_matrix.num_cols:
      identity_elements = [[1 if row == col else 0 for col in range(self.num_cols)] for row in range(self.num_rows)]
      rounded_elements = [[round(e, 11) for e in det_matrix.rref().elements[row]] for row in range(det_matrix.rref().num_rows)]
      swaps = 0
      if rounded_elements == identity_elements:
        det = 1
        row_index = 0
        for col_index in range(det_matrix.num_cols):
          pivot_row = det_matrix.get_pivot_row(col_index)
          if pivot_row != None:
            if pivot_row != row_index:
              det_matrix = det_matrix.swap_rows(row_index, pivot_row)
              swaps += 1
    
            first_index = det_matrix.nonzero_in_row(row_index)
            first_ele = det_matrix.elements[row_index][first_index]
            for col_index in range(det_matrix.num_cols):
              det_matrix.elements[row_index][col_index] *= 1/first_ele

            det_matrix = det_matrix.clear_above(row_index)
            det_matrix = det_matrix.clear_below(row_index)

            det *= first_ele

            row_index += 1
      else:
        det = 0
    else:
      return 'Error: cannot take determinant of a non-square matrix'
    return ((-1)**swaps)*det

  def exponent(self, exp):
    mult_matrix = self.copy()
    exp_matrix = self.copy()
    for i in range(exp - 1):
      exp_matrix = exp_matrix.matrix_multiply(mult_matrix)
    return exp_matrix

  def __add__(self, other_matrix):
    return self.add(other_matrix)

  def __sub__(self, other_matrix):
    return self.subtract(other_matrix)
  
  def __mul__(self, scalar):
    return self.scalar_multiply(scalar)

  def __rmul__(self, scalar):
    return self.scalar_multiply(scalar)

  def __matmul__(self, other_matrix):
    return self.matrix_multiply(other_matrix)

  def __pow__(self, exp):
    return self.exponent(exp)

  def __eq__(self, other_matrix):
    return self.elements == other_matrix.elements

  def cofactor_sub_matrix(self, row_index, col_index):
    sub_matrix = self.copy()
    row_nums = [i for i in range(sub_matrix.num_rows) if i != row_index]
    col_nums = [i for i in range(sub_matrix.num_cols) if i != col_index]
    sub_matrix = sub_matrix.get_rows(row_nums)
    sub_matrix = sub_matrix.get_columns(col_nums)
    return sub_matrix

  def cofactor_determinant(self):
    det_matrix = self.copy()
    det = 0
    if det_matrix.num_rows == det_matrix.num_cols:
      if det_matrix.num_cols > 1:
        for col_index in range(det_matrix.num_cols):
          sub_matrix = det_matrix.cofactor_sub_matrix(0,col_index)
          det += ((-1)**col_index)*det_matrix.elements[0][col_index]*sub_matrix.cofactor_determinant()
      else:
        return det_matrix.elements[0][0]
    else:
      return 'Error: cannot take determinant of a non-square matrix'
    return det

