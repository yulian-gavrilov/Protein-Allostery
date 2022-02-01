from function_call import pass_by_ref
from function_call import pyFoo
from function_call import py_sum_mat_ref
from function_call import py_ret_mat_ref
from function_call import py_dist_ref
from function_call import py_get_atom_clusters

import numpy as np

print(pass_by_ref())
print(pyFoo(1.1,2))

list5 = np.array([[1, 2, 3], [4, 5, 6]])
print("Test Sum list: ", py_sum_mat_ref(list5))

#Test7: Returning a 2D numpy array from c++
#list6 = np.array([[1, 2, 3], [4, 5, 6]])
#print("Test , Sum list: ", py_ret_mat_ref(list6))


arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(py_dist_ref(arr1,arr2))
print(type(py_dist_ref(arr1,arr2)))
