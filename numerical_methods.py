import numpy as np

class fattorizzazioni:
    '''Class that  implements some of the most known methods for solving linear system by using matrix factorization'''
    @staticmethod
    def lu_(A):
        '''Finds the matrices L & U of the LU factorization'''
        n = np.size(A,1)
        L = np.eye(n)
        U = np.zeros((n,n))
        for i in range(0,n):
            for j in range(i,n):
                U[i,j] = A[i,j] - np.dot(L[i,0:i],U[0:i,j])
            for k in range(i+1,n):
                L[k,i] = A[k,i] - np.dot(L[k,0:i],U[0:i,i])
                L[k,i]= L[k,i]/U[i,i]
        return L,U
    
    @staticmethod
    def chol_(A):
        '''Finds the matrix H' of Cholesky factorization'''
        n = np.size(A,1)
        H = np.zeros((n,n))
        for k in range(0,n):
            H[k,k] = A[k,k] - np.dot(H[k,0:k],H[k,0:k])
            H[k,k] = np.sqrt(H[k,k])
            for i in range(k+1,n):
                H[i,k] = A[i,k] - np.dot(H[i,0:k],H[k,0:k])

                H[i,k] = H[i,k]/H[k,k]
        return H

    @staticmethod
    def forwardraw(L,b):
        '''Solves the lower triangular system given a nxn-dim lower triangular matrix L and a n-dim vector b'''
        n = np.size(L,1)
        y = np.zeros((n,1))
        y[0] = b[0]/L[0,0]
        for k in range(1,n):
            l_k = L[k,0:k]
            y_k = y[0:k]
            y[k] = b[k] - np.dot(l_k,y_k)
            y[k] = y[k]/L[k,k]
        return y
    
    @staticmethod
    def backwardraw(U,y):
        '''Solves the upper triangular system given a nxn-dim upper triangular matrix U and a n-dim vector y'''
        n = np.size(U,1)
        x = np.zeros((n,1))
        x[n-1] = y[n-1]/U[n-1,n-1]
        for k in range(n-1,-1,-1):
            u_k = U[k,k+1:n]
            x_k = x[k+1:n]
            x[k] = y[k] - np.dot(u_k,x_k)
            x[k] = x[k]/U[k,k]
        return x

    @classmethod
    def lu_2(cls,A,b):
        '''Solves the system using LU factorization'''
        L,U = cls.lu_(A)
        y = cls.forwardraw(L,b)
        x = cls.backwardraw(U,y)
        return x
    
    @classmethod
    def chol_2(cls,A,b):
        '''Solves the system using Cholesky factorization'''
        H = cls.chol_(A)
        y = cls.forwardraw(H,b)
        x = cls.backwardraw(H.transpose(),y)
        return x