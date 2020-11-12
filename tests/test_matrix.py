import sys
sys.path.append('src')
from matrix import Matrix

A = Matrix([[1,3], [2,4]])
# [[1,3],
#  [2,4]]

B = A.copy()
A = 'resetting A to a string'
print('Testing method "copy"...')
assert B.elements == [[1,3], [2,4]] # the purpose of this test is to show that B is independent of A
print('PASSED')
# [[1,3],
#  [2,4]]

C = Matrix([[1,0], [2,-1]])
D = B.add(C)
print('Testing method "add"...')
assert D.elements == [[2,3], [4,3]]
print('PASSED')

E = B.subtract(C)
print('Testing method "subtract"...')
assert E.elements == [[0,3], [0,5]]
print('PASSED')

F = B.scalar_multiply(2)
print('Testing method "scalar_multiply"...')
assert F.elements == [[2,6], [4,8]]
print('PASSED')

G = B.matrix_multiply(C)
print('Testing method "matrix_multiply"...')
assert G.elements == [[7,-3], [10,-4]]
print('PASSED')

#tests for assignment 7

A = Matrix([[1,0,2,0,3],
            [0,4,0,5,0],
            [6,0,7,0,8],
            [-1,-2,-3,-4,-5]])
A_t = A.transpose()

print('Testing method "transpose"...')
assert A_t.elements == [[ 1,  0,  6, -1],
 [ 0,  4,  0, -2],
 [ 2,  0,  7, -3],
 [ 0,  5,  0, -4],
 [ 3,  0,  8, -5]]
print('PASSED')

B = A_t.matrix_multiply(A)

print('Testing general method "matrix_multiply"...')
assert B.elements == [[38,  2, 47,  4, 56],
 [ 2, 20,  6, 28, 10],
 [47,  6, 62, 12, 77],
 [ 4, 28, 12, 41, 20],
 [56, 10, 77, 20, 98]]
print('PASSED')

C = B.scalar_multiply(0.1)

print('Testing general method "scalar_multiply"...')
assert C.elements == [[3.8,  .2, 4.7,  .4, 5.6], 
 [ .2, 2.0,  .6, 2.8, 1.0], 
 [4.7,  .6, 6.2, 1.2, 7.7], 
 [ .4, 2.8, 1.2, 4.1, 2.0], 
 [5.6, 1.0, 7.7, 2.0, 9.8]], C.elements
print('PASSED')

D = B.subtract(C)

print('Testing general method "subtract"...')
assert D.elements == [[34.2,  1.8, 42.3,  3.6, 50.4], 
 [ 1.8, 18. ,  5.4, 25.2,  9. ],
 [42.3,  5.4, 55.8, 10.8, 69.3],
 [ 3.6, 25.2, 10.8, 36.9, 18. ],
 [50.4,  9. , 69.3, 18. , 88.2]]
print('PASSED')

E = D.add(C)

print('Testing general method "add"...')
assert E.elements == [[38,  2, 47,  4, 56],
 [ 2, 20,  6, 28, 10],
 [47,  6, 62, 12, 77],
 [ 4, 28, 12, 41, 20],
 [56, 10, 77, 20, 98]]
print('PASSED')

print('Testing method "is_equal"...')
assert E.is_equal(B) is True
assert E.is_equal(C) is False
print('PASSED')


#tests for assignment 8
print('Testing row reduction on the following matrix:')
print('[[0, 1, 2],')
print('[3, 6, 9],')
print('[2, 6, 8]]')

A = Matrix(elements = [[0, 1, 2],
                       [3, 6, 9],
                       [2, 6, 8]])

print('Testing method "get_pivot_row(0)"...')
assert A.get_pivot_row(0) == 1
print('PASSED')

A = A.swap_rows(0,1)
print('Testing method "swap_rows(0,1)"...')
assert A.elements == [[3, 6, 9],
                      [0, 1, 2],
                      [2, 6, 8]]
print('PASSED')

A = A.normalize_row(0)
print('Testing method "normalize_row(0)"...')
assert A.elements == [[1, 2, 3],
                      [0, 1, 2],
                      [2, 6, 8]]
print('PASSED')

A = A.clear_below(0)
print('Testing method "clear_below(0)"...')
assert A.elements == [[1, 2, 3],
                      [0, 1, 2],
                      [0, 2, 2]]
print('PASSED')

print('Testing method "get_pivot_row(1)"...')
assert A.get_pivot_row(1) == 1, A.get_pivot_row(1)
print('PASSED')

A = A.normalize_row(1)
print('Testing method "normalize_row(1)"...')
assert A.elements == [[1, 2, 3],
                      [0, 1, 2],
                      [0, 2, 2]]
print('PASSED')

A = A.clear_below(1)
print('Testing method "clear_below(1)"...')
assert A.elements == [[1, 2, 3],
                      [0, 1, 2],
                      [0, 0, -2]]
print('PASSED')

print('Testing method "get_pivot_row(2)"...')
assert A.get_pivot_row(2) == 2
print('PASSED')

A = A.normalize_row(2)
print('Testing method "normalize_row(1)"...')
assert A.elements == [[1, 2, 3],
                      [0, 1, 2],
                      [0, 0, 1]]
print('PASSED')

A = A.clear_above(2)
print('Testing method "clear_above(2)"...')
assert A.elements == [[1, 2, 0],
                      [0, 1, 0],
                      [0, 0, 1]]
print('PASSED')

A = A.clear_above(1)
print('Testing method "clear_above(1)"...')
assert A.elements == [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]
print('PASSED')

#Assignment 11 tests

A = Matrix([[0, 1, 2],
            [3, 6, 9],
            [2, 6, 8]])
print('Testing rref on the following matrix:')
print('[[0, 1, 2],')
print('[3, 6, 9],')
print('[2, 6, 8]]')
assert A.rref().elements == [[1, 0, 0], [0, 1, 0], [0, 0, 1]], str(A.rref().elements) 
print('PASSED')


B = Matrix([[0, 0, -4, 0],
            [0, 0, 0.3, 0],
            [0, 2, 1, 0]])
print('Testing rref on the following matrix:')
print('[[0, 0, -4, 0],')
print('[0, 0, 0.3, 0],')
print('[0, 2, 1, 0]]')

assert B.rref().elements == [[0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0]], B.rref().elements
print('PASSED')

#Assignment 15

A = Matrix([
    [1, 2,   3,  4],
    [5, 6,   7,  8],
    [9, 10, 11, 12]
])
B = Matrix([
    [13, 14],
    [15, 16],
    [17, 18]
])

A_augmented = A.augment(B)
print('Testing method "augment"...')
assert A_augmented.elements == [[1, 2,   3,  4, 13, 14],
                                [5, 6,   7,  8, 15, 16],
                                [9, 10, 11, 12, 17, 18]]
print('PASSED')

print('Testing method "get_rows"...')
rows_02 = A_augmented.get_rows([0, 2])
assert rows_02.elements == [[1, 2,   3,  4, 13, 14],
                            [9, 10, 11, 12, 17, 18]]
print('PASSED')

print('Testing method "get_columns"...')
cols_0123 = A_augmented.get_columns([0, 1, 2, 3])
assert cols_0123.elements == [[1, 2,   3,  4],
                              [5, 6,   7,  8],
                              [9, 10, 11, 12]]

cols_45 = A_augmented.get_columns([4, 5])
assert cols_45.elements == [[13, 14],
                            [15, 16],
                            [17, 18]]
print('PASSED')

#Assignment 17

A = Matrix([[1, 2],
            [3, 4]])
A_inv = A.inverse()
print('Testing method "inverse"...')
assert A_inv.elements == [[-2,   1],
                          [1.5, -0.5]]

A = Matrix([[1,   2,  3],
            [1,   0, -1],
            [0.5, 0,  0]])
A_inv = A.inverse()
assert A_inv.elements == [[0,   0,    2],
                          [0.5, 1.5, -4],
                          [0,  -1,    2]]

A = Matrix([[1, 2, 3, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0]])
A_inv = A.inverse()
assert A_inv == "Error: cannot invert a non-square matrix"

A = Matrix([[1, 2, 3],
            [3, 2, 1],
            [1, 1, 1]])
A_inv = A.inverse()
assert A_inv == "Error: cannot invert a singular matrix"

print('PASSED')

#Assignment 21

print('Testing method "determinant"...')
A = Matrix(elements = [[1,2],
                       [3,4]])
ans = A.determinant()
assert round(ans,6) == -2

A = Matrix(elements = [[1,2,0.5],
                       [3,4,-1],
                       [8,7,-2]])
ans = A.determinant()
assert round(ans,6) == -10.5

A = Matrix(elements = [[1,2,0.5,0,1,0],
                       [3,4,-1,1,0,1],
                       [8,7,-2,1,1,1],
                       [-1,1,0,1,0,1],
                       [0,0.35,0,-5,1,1],
                       [1,1,1,1,1,0]])
ans = A.determinant()
assert round(ans,6) == -37.3

A = Matrix(elements = [[1,2,0.5,0,1,0],
                       [3,4,-1,1,0,1],
                       [8,7,-2,1,1,1],
                       [-1,1,0,1,0,1],
                       [0,0.35,0,-5,1,1],
                       [1,1,1,1,1,0],
                       [2,3,1.5,1,2,0]])
ans = A.determinant()
assert ans == 'Error: cannot take determinant of a non-square matrix'

A = Matrix(elements = [[1,2,0.5,0,1,0,1],
                       [3,4,-1,1,0,1,0],
                       [8,7,-2,1,1,1,0],
                       [-1,1,0,1,0,1,0],
                       [0,0.35,0,-5,1,1,0],
                       [1,1,1,1,1,0,0],
                       [2,3,1.5,1,2,0,1]])
ans = A.determinant()
assert round(ans,6) == 0
print('PASSED')

#Assignment 26

A = Matrix([[1, 1, 0],
            [2, -1, 0],
            [0, 0, 3]])
print('Testing method "exponent"...')
assert A.exponent(3).elements == [[3, 3, 0],
                                  [6, -3, 0],
                                  [0, 0, 27]]
print('PASSED')

A = Matrix(
    [[1,0,2,0,3],
    [0,4,0,5,0],
    [6,0,7,0,8],
    [-1,-2,-3,-4,-5]]
    )
print('Testing overloaded operations...')
A_t = A.transpose()
assert A_t.elements == [[ 1,  0,  6, -1],
                        [ 0,  4,  0, -2],
                        [ 2,  0,  7, -3],
                        [ 0,  5,  0, -4],
                        [ 3,  0,  8, -5]]

B = A_t @ A
assert B.elements == [[38,  2, 47,  4, 56],
                      [ 2, 20,  6, 28, 10],
                      [47,  6, 62, 12, 77],
                      [ 4, 28, 12, 41, 20],
                      [56, 10, 77, 20, 98]]

C = B * 0.1
assert C.elements == [[3.8,  .2, 4.7,  .4, 5.6],
                      [ .2, 2.0,  .6, 2.8, 1.0],
                      [4.7,  .6, 6.2, 1.2, 7.7],
                      [ .4, 2.8, 1.2, 4.1, 2.0],
                      [5.6, 1.0, 7.7, 2.0, 9.8]]

D = B - C
assert D.elements == [[34.2,  1.8, 42.3,  3.6, 50.4],
                      [ 1.8, 18. ,  5.4, 25.2,  9. ],
                      [42.3,  5.4, 55.8, 10.8, 69.3],
                      [ 3.6, 25.2, 10.8, 36.9, 18. ],
                      [50.4,  9. , 69.3, 18. , 88.2]]

E = D + C
assert E.elements == [[38,  2, 47,  4, 56],
                      [ 2, 20,  6, 28, 10],
                      [47,  6, 62, 12, 77],
                      [ 4, 28, 12, 41, 20],
                      [56, 10, 77, 20, 98]]

assert (E == B) == True

assert (E == C) == False
print('PASSED')

print('Testing method "cofactor_determinant"...')
A = Matrix(elements = [[1,2],
                       [3,4]])
ans = A.cofactor_determinant()
assert round(ans,6) == -2

A = Matrix(elements = [[1,2,0.5],
                       [3,4,-1],
                       [8,7,-2]])
ans = A.cofactor_determinant()
assert round(ans,6) == -10.5

A = Matrix(elements = [[1,2,0.5,0,1,0],
                       [3,4,-1,1,0,1],
                       [8,7,-2,1,1,1],
                       [-1,1,0,1,0,1],
                       [0,0.35,0,-5,1,1],
                       [1,1,1,1,1,0]])
ans = A.cofactor_determinant()
assert round(ans,6) == -37.3

A = Matrix(elements = [[1,2,0.5,0,1,0],
                       [3,4,-1,1,0,1],
                       [8,7,-2,1,1,1],
                       [-1,1,0,1,0,1],
                       [0,0.35,0,-5,1,1],
                       [1,1,1,1,1,0],
                       [2,3,1.5,1,2,0]])
ans = A.cofactor_determinant()
assert ans == 'Error: cannot take determinant of a non-square matrix'

A = Matrix(elements = [[1,2,0.5,0,1,0,1],
                       [3,4,-1,1,0,1,0],
                       [8,7,-2,1,1,1,0],
                       [-1,1,0,1,0,1,0],
                       [0,0.35,0,-5,1,1,0],
                       [1,1,1,1,1,0,0],
                       [2,3,1.5,1,2,0,1]])
ans = A.cofactor_determinant()
assert round(ans,6) == 0
print('PASSED')

#Assignment 27

A = Matrix([[1, 1, 0],
                [2, -1, 0],
                [0, 0, 3]])
print('Testing other overloaded operations...')
B = 0.1 * A
assert B.elements == [[0.1, 0.1, 0],
 [0.2, -0.1, 0],
 [0, 0, 0.3]]

C = A**3
assert C.elements == [[3, 3, 0],
 [6, -3, 0],
 [0, 0, 27]]
print('PASSED')

