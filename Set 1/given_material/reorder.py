import numpy as np
from scipy import sparse

def bfs_levelset(C, V):
 U = np.arange(0,np.size(C,1))
 P = np.array([],dtype=int)
 # print('V:'); print(V)
 U[V] = -1; # print('U:'); print(U)

 while np.size(V) > 0:
  P = np.append(P,V); # print('P:'); print(P)
  L = V; # print('L:'); print(L)
  V = np.array([],dtype=int); # print('V:'); print(V)
  for v in L:
   # assume directed neighbors is row
   # nonzero returns (I,J) with
   # I = [0,0,...] and J = [neighbor numbers]
   N = C[v,:]; N = np.nonzero(N)[1]; # print('N:'); print(N)
   # remove already visit neighbors (for which U(.) = 0)
   N = N[U[N] > -1]; # print('N:'); print(N)
   # stack found neighbors
   V = np.append(V,N); # print('V:'); print(V)
   # mark found vertices as visits
   U[N] = -1; # print('U:'); print(U)
 return P


def bfs_queue(C, V):
 U = np.arange(0,np.size(C,1))
 P = np.array([V],dtype=int)
 # print('V:'); print(V)
 # print('P:'); print(P)
 U[V] = -1; # print('U:'); print(U)
 i = 0;

 while i < len(P):
  v = P[i];
  N = C[v,:]; N = np.nonzero(N)[1]; # print('N:'); print(N)
  # remove already visit neighbors (for which U(.) = 0)
  N = N[U[N] > -1]; # print('N:'); print(N)
  # stack found neighbors
  P = np.append(P,N); # print('V:'); print(V)
  # mark found vertices as visits
  U[N] = -1; # print('U:'); print(U)
  i += 1
 return P


def bfs_levelset_greedy(C, V):
 U = np.arange(0,np.size(C,1))
 P = np.array([V],dtype=int)
 # print('V:'); print(V)
 U[V] = -1; # print('U:'); print(U)

 while np.size(V) > 0:
  # assume directed neighbors is row
  # nonzero returns (I,J) with
  # I = [0,0,...] and J = [neighbor numbers]
  N = sparse.csc_matrix.sum(C[V,:],0); N = np.nonzero(N)[1]; print('N:'); print(N)
  # remove already visit neighbors (for which U(.) = 0)
  N = N[U[N] > -1]; # print('N:'); print(N)
  # stack found neighbors
  V = N; # print('V:'); print(V)
  # mark found vertices as visits
  U[N] = -1; # print('U:'); print(U)
  P = np.append(P,V); # print('P:'); print(P)
 return P


# random n x n connectivity matrix with maximal k non-zero entries
# (potentially not singly connected graph so P could fail)
def rcm(n,k):
 # n: amount of vertices
 # k: amount of arcs
 # double arcs permitted -- random interval is [low,high) ...
 I = np.random.randint(low=0, high=n, size=k); # print(I)
 J = np.random.randint(low=0, high=n, size=k); # print(J)
 V = np.ones(k,dtype=int); # print(V)
 C = sparse.csc_matrix((V,(I, J)), shape=(n,n))
 C.data.fill(1)
 return C

# specific nxn connectivity matrices (singly connected graphs)
def scm(select = 0):
 if select == 0:
  I = np.array([3,0,4,5,2,2,4,3,5,3]);
  J = np.array([3,4,1,0,1,3,5,4,2,5]);
  V = np.array([1,1,1,1,1,1,1,1,1,1]);
  C = sparse.csc_matrix((V,(I, J)), shape=(6,6))
  C.data.fill(1)
  return C
 elif select == 1:
  I = np.array([3,0,4,5,2,8,5,4,0,3,6,4,1,7,3,0,8,9]);
  J = np.array([8,4,1,0,1,4,2,5,8,7,7,6,3,9,2,0,1,3]);
  V = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]);
  C = sparse.csc_matrix((V,(I, J)), shape=(10,10))
  C.data.fill(1)
  return C

# 10 x 10 connectivity matrix, vertices 0 ... 9, start with first vertex
#np.random.seed(1234);
C = rcm(20,80); print('C:'); print(C.todense())
V = np.array([0],dtype=int);
P = bfs_levelset(C,V); print('Found permutation:'); print(P+1)
P = bfs_queue(C,V); print('Found permutation:'); print(P+1)
P = bfs_levelset_greedy(C,V); print('Found permutation:'); print(P+1)
# 1-st specfic connectivity matrix, start with first vertex
C = scm(); print('C:'); print(C.todense());
V = np.array([0],dtype=int);
P = bfs_levelset(C,V); print('Found permutation:'); print(P+1)
P = bfs_queue(C,V); print('Found permutation:'); print(P+1)
P = bfs_levelset_greedy(C,V); print('Found permutation:'); print(P+1)
# 2-nd specfic connectivity matrix, start with first vertex
C = scm(1); print('C:'); print(C.todense());
V = np.array([0],dtype=int);
P = bfs_levelset(C,V); print('Found permutation:'); print(P+1)
P = bfs_queue(C,V); print('Found permutation:'); print(P+1)
P = bfs_levelset_greedy(C,V); print('Found permutation:'); print(P+1)

