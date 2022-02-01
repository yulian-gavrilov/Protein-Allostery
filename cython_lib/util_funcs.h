#ifndef SETINT_H
#define SETINT_H

#include <vector>
#include <iostream>
#include <math.h>

void set_integer_ref(int& x);
double cppFoo(double x, int y);
double cpp_sum_mat_ref(const std::vector< std::vector<double> > & sv);
std::vector< std::vector<double> > cpp_ret_mat_ref(const std::vector< std::vector<double> > & sv);
std::vector< std::vector<double> > cpp_dist_ref(const std::vector< std::vector<double> > & vec1, const std::vector< std::vector<double> > & vec2);
std::vector< std::vector<int> > cpp_get_atom_clusters(const std::vector< std::vector<int> > & grid_clusters, const std::vector< int > & gridPoints2atoms_ndx);

#endif

