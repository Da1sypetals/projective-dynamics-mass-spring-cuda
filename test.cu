//
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/copy.h>
//#include <iostream>
//#include <cusparse.h>
//#include "MatrixTypes.hpp"
//#include <Eigen/Sparse>
//#include "Types.hpp"
//
//int main() {
//    // Initialize cuSPARSE
//    cusparseHandle_t handle = nullptr;
//    cusparseCreate(&handle);
//
//    eg::SparseMatrix<float, eg::RowMajor> h_mat(3, 3);
//
//    std::vector<eg::Triplet<float>> tr;
//    tr.push_back(eg::Triplet<float>(0, 0, 1));
//    tr.push_back(eg::Triplet<float>(0, 1, 3));
//    tr.push_back(eg::Triplet<float>(0, 2, 4));
//    tr.push_back(eg::Triplet<float>(1, 0, 0));
//    tr.push_back(eg::Triplet<float>(1, 1, 2));
//    tr.push_back(eg::Triplet<float>(1, 2, 6));
//    tr.push_back(eg::Triplet<float>(2, 0, 3));
//    tr.push_back(eg::Triplet<float>(2, 1, 6));
//    tr.push_back(eg::Triplet<float>(2, 2, 2));
//    h_mat.setFromTriplets(tr.begin(), tr.end());
//
//    csr_rect d_mat(h_mat);
//
//    // Initialize your dense 3-vectors on the device
//    thrust::device_vector<float> d_x_vec = {7, 8, 9};  // Initialize vector x
//    dense_vec d_x(d_x_vec);
//    thrust::device_vector<float> d_y_vec = {1, 3, 1};  // Initialize vector y
//    dense_vec d_y(d_y_vec);
//
//    // Define scalar constants for the operation
//    float alpha = 1.0f;
//    float beta = 1.5f;
//
//    // Perform Axpby operation
//    d_mat.Axpby(handle, alpha, d_x, beta, d_y);
//
//    // Transfer the result from the device to the host
//    thrust::host_vector<float> h_y_vec(d_y_vec.size());
//    thrust::copy(d_y_vec.begin(), d_y_vec.end(), h_y_vec.begin());
//
//    // Print out the results
//    std::cout << "The resulting vector is: [ ";
//    for (const auto &val: h_y_vec) {
//        std::cout << val << " ";
//    }
//    std::cout << "]" << std::endl;
//
//    // Destroy cuSPARSE handle
//    cusparseDestroy(handle);
//
//    return 0;
//}
