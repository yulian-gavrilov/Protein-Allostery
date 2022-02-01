
from libcpp.vector cimport vector

cdef extern from "util_funcs.h":
    void set_integer_ref(int&)

cpdef pass_by_ref():
    cdef int x
    set_integer_ref(x)
    return x


cdef extern from "util_funcs.h":
    double cppFoo(double x, int y)

cpdef pyFoo(double x, int y):
    return cppFoo(x, y)


cdef extern from "util_funcs.h":
    double cpp_sum_mat_ref(vector[vector[double]] &)

cpdef py_sum_mat_ref(sv):
    return cpp_sum_mat_ref(sv)



cdef extern from "util_funcs.h":
    vector[vector[double]] cpp_ret_mat_ref(vector[vector[double]] &)

cpdef py_ret_mat_ref(sv):
    return cpp_ret_mat_ref(sv)



cdef extern from "util_funcs.h":
    vector[vector[double]] cpp_dist_ref(vector[vector[double]] &, vector[vector[double]] &)

cpdef py_dist_ref(vec1,vec2):
    return cpp_dist_ref(vec1,vec2)


cdef extern from "util_funcs.h":
    vector[vector[int]] cpp_get_atom_clusters(vector[vector[int]] &, vector[int] &)

cpdef py_get_atom_clusters(vec1,vec2):
    return cpp_get_atom_clusters(vec1,vec2)


