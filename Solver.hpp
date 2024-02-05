#include <vector>
#include <Eigen/Sparse>
#include "Types.hpp"


struct CSRMatrix {
    int n;
    int nnz;
    thrust::device_vector<int> rowPtr;
    thrust::device_vector<int> colIdx;
    thrust::device_vector<float> val;

};


class Solver {

    

    int n; // number of vertex
    std::vector<Constraint> constraints;

    
    
    


};