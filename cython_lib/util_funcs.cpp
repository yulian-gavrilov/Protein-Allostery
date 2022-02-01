#include "util_funcs.h"

void set_integer_ref(int& x)
{
    x = 100;
}

double cppFoo(double x, int y)
{
    return 2 * x + y;
}


/*
Passing a 2D vector by reference or list and returning its sum
*/

double cpp_sum_mat_ref(const std::vector< std::vector<double> > & sv)
{

double tot=0;

int svrows = sv.size();
int svcols = sv[0].size();
std::cout << "vector length " << svrows << " , " << svcols << std::endl;

for (int ii=0; ii<svrows; ii++)
{
        for (int jj=0; jj<svcols; jj++)
        {
                tot = tot + sv.at(ii).at(jj);
        }
}        
return tot;

}



/*
Inputting a 2D vector, performing a simple operation and returning a new 2D vector
*/
std::vector< std::vector<double> > cpp_ret_mat_ref(const std::vector< std::vector<double> > & sv)
{

int svrows = sv.size();
int svcols = sv[0].size();

std::vector< std::vector<double> > tot;
tot.resize(svrows, std::vector<double> (svcols, -1));


std::cout << "vector length " << svrows << " , " << svcols << std::endl;

for (int ii=0; ii<svrows; ii++)
{
        for (int jj=0; jj<svcols; jj++)
        {
                tot.at(ii).at(jj) = (2*sv.at(ii).at(jj));
        }
}        
return tot;

}


/*
Inputting two 2D vectors: 1d: [x,y,z] for 2d: points. Calculating the distance between the points and returning a new 2D vector 
*/
std::vector< std::vector<double> > cpp_dist_ref(const std::vector< std::vector<double> > & vec1, const std::vector< std::vector<double> > & vec2)
{

    int vec1_rows = vec1.size();
    //int vec1_cols = vec1[0].size();

    int vec2_rows = vec2.size();
    //int vec2_cols = vec2[0].size();

    std::vector< std::vector<double> > dm;
    dm.resize(vec1_rows, std::vector<double> (vec2_rows, -1));

    std::cout << "vector length " << vec1_rows << " , " << vec2_rows << std::endl;
    
    for (int ii=0; ii<vec1_rows; ii++)
    {
        for (int jj=0; jj<vec2_rows; jj++)
        {
            dm[ii][jj] = pow(pow(vec1[ii][0]-vec2[jj][0],2) + pow(vec1[ii][1]-vec2[jj][1],2) + pow(vec1[ii][2]-vec2[jj][2],2),0.5);
        }
    }
    
    return dm;

}


/*
get atom clusters from grid clusters based on indices in gridPoints2atoms_ndx 
*/
std::vector< std::vector<int> > cpp_get_atom_clusters(const std::vector< std::vector<int> > & grid_clusters, const std::vector< int > & gridPoints2atoms_ndx)
{

    std::vector< std::vector<int> > atom_clusters = grid_clusters;

    for (size_t i=0; i<grid_clusters.size(); i++)
    {
        for (size_t j=0; j<grid_clusters.at(i).size(); j++)
        {
            int index = (int)round(grid_clusters.at(i).at(j))-1;
            atom_clusters.at(i).at(j) = gridPoints2atoms_ndx.at(index);
        }
    }

    return atom_clusters;
}
