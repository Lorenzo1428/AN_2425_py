import numpy as np
import copy
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
    def lu_band(A,p,q):
        '''Finds the LU factorization optimized for band matrices'''
        n = np.size(A,1)
        L = np.eye(n)
        U = np.zeros((n,n))
        for i in range(0,n):
            for j in range(i,min(i+p+1,n)):
                U[i,j] = A[i,j] - np.dot(L[i,max(0,i-q):i],U[max(0,i-p):i,j])
            for k in range(i+1,min(i+q+1,n)):
                L[k,i] = A[k,i] - np.dot(L[k,max(0,k-q):i],U[max(0,k-p):i,i])
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
    def chol_band(A,q):
        '''Finds the Cholesky factorization optimized for band matrices'''
        n = np.size(A,1)
        H = np.zeros((n,n))
        for k in range(0,n):
            H[k,k] = A[k,k] - np.dot(H[k,max(0,k-q):k],H[k,max(0,k-q):k])
            H[k,k] = np.sqrt(H[k,k])
            for i in range(k+1,min(n,k+q+1)):
                H[i,k] = A[i,k] - np.dot(H[i,max(0,k-q):k],H[k,max(0,k-q):k])

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

    @staticmethod
    def forwband(L,b,q):
        '''Solves the lower triangular system given a nxn-dim band lower triangular matrix L and a n-dim vector b'''
        n = np.size(L,1)
        y = np.zeros((n,1))
        y[0] = b[0]/L[0,0]
        for k in range(1,n):
            l_k = L[k,max(0,k-q):k]
            y_k = y[max(0,k-q):k]
            y[k] = b[k] - np.dot(l_k,y_k)
            y[k] = y[k]/L[k,k]
        return y

    @staticmethod
    def backband(U,y,p):
        '''Solves the upper triangular system given a nxn-dim band upper triangular matrix U and a n-dim vector y'''
        n = np.size(U,1)
        x = np.zeros((n,1))
        x[n-1] = y[n-1]/U[n-1,n-1]
        for k in range(n-1,-1,-1):
            u_k = U[k,k+1:min(n,k+p)]
            x_k = x[k+1:min(n,k+p)]
            x[k] = y[k] - np.dot(u_k,x_k)
            x[k] = x[k]/U[k,k]
        return x    
    
    @staticmethod
    def gs_mod(A):
        copyA = copy.copy(A)
        m = np.size(copyA,0)
        n = np.size(copyA,1)
        Q = np.zeros((m,n))
        R = np.zeros((n,n))
        for k in range(0,n):
            Ak = copyA[:,k]
            r = np.linalg.norm(Ak)
            Q[:,k] = Ak/r 
            R[k,k] = r
            Qk = Q[:,k]
            for j in range(k+1,n):
                Aj = copyA[:,j]
                r = np.dot(Qk,Aj)
                copyA[:,j] = Aj - r*Qk
                R[k,j] = r;      
        return Q,R
    
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
    