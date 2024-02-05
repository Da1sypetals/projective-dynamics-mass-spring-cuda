#pragma once

#include "MatrixTypes.hpp"
#include "thrust/host_vector.h"
#include "kernels.cuh"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

struct Axpby {
    float a, b;

    Axpby(float _a, float _b) : a(_a), b(_b) {}

    __host__ __device__
    float operator()(const float &x, const float &y) const {
        return a * x + b * y;
    }
};

struct GlobalAxpy {
    __host__ __device__
    float operator()(const float &x, const float &y) {
        return x + dt2 * y;
    }
};

struct AdvanceAxpy {
    __host__ __device__
    float operator()(const float &x, const float &y) {
        return (2 - preservation) * x + (1 - preservation) * y;
    }
};


class D_Solver {

    friend class D_Preprocessor;

private:

    bool initialized;

    GlobalAxpy global_axpy;
    AdvanceAxpy advance_axpy;

public:

    cusolverSpHandle_t cusolverHandle; // in-class
    cusparseHandle_t cusparseHandle; // in-class

    int n_iter; // ctor

    std::shared_ptr<Cloth> cloth; // ctor

    eg::SparseMatrix<float, eg::RowMajor> h_Y; // pre
    eg::SparseMatrix<float, eg::RowMajor> h_M; // pre
    eg::SparseMatrix<float, eg::RowMajor> h_L; // pre
    eg::SparseMatrix<float, eg::RowMajor> h_J; // pre


    thrust::host_vector<float> h_x; // pre


    thrust::device_vector<float> d_d; // pre
    thrust::device_vector<float> d_x_prev; // pre
    thrust::device_vector<float> d_x; // pre
    thrust::device_vector<float> d_y; // pre
    thrust::device_vector<float> d_b; // pre
    thrust::device_vector<float> d_f_external; // pre


    dense_vec dn_d; // pre
    dense_vec dn_b; // pre


    csr_matrix d_Y; // pre
    csr_matrix d_M; // pre
    csr_matrix d_L; // pre
    csr_rect d_J; // pre

    std::unique_ptr<D_Cholesky> cholesky; // pre
    thrust::device_vector<Constraint> d_constraints; // pre

    std::vector<std::pair<int, float3>> fixed;


    int index(int irow, int icol) {
        return irow * cloth->nside + icol;
    }

    void GlobalStep() {


        // (1) b = y + h2 * f_ext
        thrust::transform(d_y.begin(), d_y.end(), d_f_external.begin(), d_b.begin(), global_axpy);

        // (2) Axpby(handle, h2, d, 1, b) => b = h2 * J * d + b
        d_J.Axpby(cusparseHandle, dt2, dn_d, 1, dn_b);

        // (1) (2) b = h2*J*d + y + h2*f_ext

        // (3) solve linear system Yx = b
        cholesky->Solve(cusolverHandle,
                        thrust::raw_pointer_cast(d_b.data()),
                        thrust::raw_pointer_cast(d_x.data()));


    }

    void LocalStep() {
        D_LocalStep<<<64, 32>>>(cloth->numConstraint,
                                d_constraints.data(),
                                d_x.data(),
                                d_d.data());
        cudaDeviceSynchronize();
        std::cout << "d: " << std::endl;
        for (int i = 0; i < cloth->numConstraint; i++) {
            printf("[%.3f, %.3f, %.3f]  \n", d_d[3 * i], d_d[3 * i + 1], d_d[3 * i + 2]);
        }
        std::cout << std::endl << std::endl;
    }


    // API
    D_Solver(std::shared_ptr<Cloth> _cloth, int _n_iter)
            : cloth(_cloth), n_iter(_n_iter) {

        d_constraints.resize(3 * cloth->numConstraint);
        thrust::copy(cloth->constraints.begin(), cloth->constraints.end(), d_constraints.begin());

    }

    void SetHandles(cusolverSpHandle_t _cusolverHandle, cusparseHandle_t _cusparseHandle) {
        cusolverHandle = _cusolverHandle;
        cusparseHandle = _cusparseHandle;
    }

    void AddFixed(int irow, int icol) {

        if (not initialized) {
            std::cerr << "Fatal: Not initialized, cannot add fixed vertex!" << std::endl;
            exit(1);
        }

        int idx = index(irow, icol);
        fixed.push_back(std::make_pair(index(irow, icol),
                                       float3{d_x[3 * idx], d_x[3 * idx + 1], d_x[3 * idx + 2]}));
    }

    void Step() {

        thrust::transform(d_x.begin(),
                          d_x.end(),
                          d_x_prev.begin(),
                          d_y.begin(),
                          advance_axpy);
        std::cout << "transform\n";


        thrust::copy(d_x.begin(), d_x.end(), d_x_prev.begin());
        std::cout << "copy prev\n";


        for (int iter = 0; iter < n_iter; iter++) {
            LocalStep();
            GlobalStep();

        }

        for (auto &&[ifixed, fixpos]: fixed) {
            d_x[3 * ifixed] = fixpos.x;
            d_x[3 * ifixed + 1] = fixpos.y;
            d_x[3 * ifixed + 2] = fixpos.z;
        }

        thrust::copy(d_x.begin(), d_x.end(), h_x.begin());
        std::cout << "copy to host\n";


    }


};








