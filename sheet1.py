import numpy as np
import time 
from numerical_methods import fattorizzazioni as f


m = 500
n = 400
A = np.random.rand(m,n)

Q,R = f.gs_mod(A)
A1 = np.matmul(Q,R)
print( np.linalg.norm(np.matmul(Q,R) - A, np.inf))