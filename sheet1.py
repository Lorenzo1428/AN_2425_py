import numpy as np
from numerical_methods import fattorizzazioni as f
n = 3
A = np.random.rand(n,n)
A = np.matmul(A,A.transpose())
#A =  2*np.eye(n) - (np.diag(np.ones((n-1)),-1) + np.diag(np.ones((n-1)), +1))
b = np.random.rand(n,1)
H = f.chol_(A)
x = f.chol_2(A,b)
H1 = np.linalg.cholesky(A)
print(np.linalg.norm(np.matmul(A,x) - b, np.inf))