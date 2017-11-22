import restore_mat
from numpy import *

mat_name = 'user_group'

inf = open(mat_name+".txt",'r')

i = 0
mat = []
while (1):
    a = inf.readline().strip('\n').split(" ")
    if (a != [""]):
        vec = []
        for i in range(len(a)):
            vec.append(int(a[i]))
        mat.append(vec)
    else:
        break

mat = array(mat)

restore_mat.store_submat(0,mat[0:10], mat_name)
restore_mat.store_submat(10,mat[10:],mat_name)


m1 = restore_mat.get_submat(0, mat_name)
m2 = restore_mat.get_submat(10, mat_name)
print(f'm1 = {m1}')
print(f'm2 = {m2}')