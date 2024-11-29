import numpy as np
from numerical_methods import fattorizzazioni as f
n = 3
A = np.random.rand(n,n)
A = np.matmul(A,A.transpose())
b = np.random.rand(n,1)
x = f.lu_2(A,b)
x1 = np.linalg.solve(A,b)
print(np.linalg.norm(x - x1, np.inf))