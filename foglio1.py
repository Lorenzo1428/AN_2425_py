import numpy as np
from metodi_numerici import fattorizzazioni as f
n = 3
A = np.random.rand(n,n)
A = np.matmul(A,A.transpose())
L,U = f.lu_(A)
print(np.linalg.norm( np.matmul(L,U) - A , np.inf) )
