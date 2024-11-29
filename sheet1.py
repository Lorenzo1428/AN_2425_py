import numpy as np
import time 
from numerical_methods import fattorizzazioni as f


n = 500
A = np.random.rand(n,n)
A = np.matmul(A,A.transpose())
A =  2*np.eye(n) - (np.diag(np.ones((n-1)),-1) + np.diag(np.ones((n-1)), +1))
b = np.random.rand(n,1)
t = time.time()
H = f.chol_(A)
print("elapsed time: ", time.time() - t )
t = time.time()
Hb = f.chol_band(A,1)
print("elapsed time: ", time.time() - t )
print(np.linalg.norm(H - Hb, np.inf))