import sys
sys.path.append('src')
from matrix import Matrix
import random

mat = Matrix([[3*random.random()-6 for i in range(10)] for j in range(10)])

#print(mat.elements)

print("RREF determinant", mat.determinant())
print("Cofactor determinant", mat.cofactor_determinant())

#The rref method is faster, because it has to go through fewer operations than the cofactor method.