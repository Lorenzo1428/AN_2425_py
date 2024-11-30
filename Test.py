import numpy as np

m = 3
n = 2
A = np.random.rand(m,n)
m = np.size(A,1)
Ak = A[:,1]/3
print(Ak)