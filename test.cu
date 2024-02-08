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




__global__ void kernel(int n, thrust::device_ptr<float> a, thrust::device_ptr<float> p) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > n) {
        return;
    }

    p[tid] = a[tid] * a[tid];

}


//int main() {
//
//    int n;
//    float init, step;
//    std::cout << "N, init, step:" << std::endl;
//    std::cin >> n >> init >> step;
//
//    thrust::device_vector<float> a(n);
//    thrust::device_vector<float> p(n);
//
//
//    thrust::sequence(a.begin(), a.end(), init, step);
//
//    kernel<<<32, 16>>>(n, a.data(), p.data());
//
//    for (int i = 0; i < n; i++) {
//        std::cout << p[i] << " ";
//    }
//
//    return 0;
//
//}



//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <cusolverSp.h>
//#include <Eigen/Sparse>
//#include <Eigen/Dense>
//#include "MatrixTypes.hpp"
//#include <iostream>
//
//int main() {
//    cusolverSpHandle_t handle = nullptr;
//    cusolverSpCreate(&handle);
//
//    // Define matrix entries
//    // (row, column, value), indexing starts at 0
//    typedef Eigen::Triplet<float> T;
//    std::vector<T> tripletList;
//    tripletList.reserve(9);
//    tripletList.push_back(T(0, 0, 4));
//    tripletList.push_back(T(0, 1, 1));
//    tripletList.push_back(T(0, 2, 1));
//    tripletList.push_back(T(1, 0, 1));
//    tripletList.push_back(T(1, 1, 3));
//    tripletList.push_back(T(1, 2, 1));
//    tripletList.push_back(T(2, 0, 1));
//    tripletList.push_back(T(2, 1, 1));
//    tripletList.push_back(T(2, 2, 2));
//
//    // Create the Eigen sparse matrix
//    Eigen::SparseMatrix<float, Eigen::RowMajor> spA(3,3);
//    spA.setFromTriplets(tripletList.begin(), tripletList.end());
//
//    // b is a 3x1 vector
//    Eigen::Vector3f b;
//    b << 1, 2, 3;
//
//
//    // Create csr_matrix from spA
//    csr_matrix csr_A(spA);
//
//    // Initialize the Cholesky solver
//    D_Cholesky solver(csr_A);
//
//    // Buffers to hold the solution and right-hand side vector
//    thrust::device_vector<float> d_x_vec = {0, 0, 0};
//    float *d_x = thrust::raw_pointer_cast(d_x_vec.data());
//    thrust::device_vector<float> d_b_vec = {b(0), b(1), b(2)};
//    float *d_b = thrust::raw_pointer_cast(d_b_vec.data());
//
//    // Solve the system Ax = b
//    int singularity = solver.Solve(handle, d_b, d_x);
//
//    // Copy the solution to the host
//    thrust::host_vector<float> h_x_vec = d_x_vec;
//
//    // Print out the results
//    std::cout << "The solution vector is: [ ";
//    for (const auto &val: h_x_vec) {
//        std::cout << val << " ";
//    }
//    std::cout << "]" << std::endl;
//
//    std::cout << "Singularity: " << singularity << std::endl;
//
//    // Destroy cusolverSp handle
//    cusolverSpDestroy(handle);
//
//    return 0;
//}


