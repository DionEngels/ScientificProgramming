import numpy as np
import numpy.matlib as ml

def square_triangular_grid(n):
 # I is lower left vertex of for every of the n^2 sub-squares of [0,1]
 I = np.arange(0,n*n-n,dtype=int)
 D = n*np.arange(1,n,dtype=int)-1
 I = np.delete(I,np.hstack((D,D)),axis=0)
 CL = np.vstack((np.column_stack((I,I+1,I+n)), np.column_stack((I+1,I+n,I+n+1))));
 #print('CL:'); print(CL)

 X = ml.repmat(np.arange(0,n).reshape((n,1))/(n-1),n,1).reshape(n*n,1);
 Y = ml.repmat(np.arange(0,n).reshape((n,1))/(n-1),1,n).reshape(n*n,1);
 XYZ = np.hstack((X,Y))
 #print(XYZ)

 return XYZ,CL
