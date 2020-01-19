from numpy import *

A = array([[1,2,3],[4,5,6],[7,8,9]])
print(A)
coords = where(A%2==0)
print(coords)
i, j = where(A%2==0)
B=array([[6,6,6],[4,4,4],[2,2,2]])
i, j = where((A>3)&(B<5)>0)
print(i,j)
