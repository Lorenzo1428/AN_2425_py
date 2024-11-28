import numpy as np

class fattorizzazioni:
    @staticmethod
    def lu_(A):
        n = np.size(A,1)
        n = 3
        L = np.eye(n)
        U = np.zeros((n,n))
        for i in range(0,n):
            for j in range(i,n):
                U[i,j] = A[i,j] - np.dot(L[i,0:i],U[0:i,j])
            for k in range(i+1,n):
                L[k,i] = A[k,i] - np.dot(L[k,0:i],U[0:i,i])
                L[k,i]= L[k,i]/U[i,i]
        return L,U