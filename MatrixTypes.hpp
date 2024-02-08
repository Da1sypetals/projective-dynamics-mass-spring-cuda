#pragma once

#include "thrust/device_vector.h"
#include "cusparse_v2.h"
#include "cusolverSp.h"
#include "Types.hpp"
#include <iostream>

struct dense_vec {

    cusparseDnVecDescr_t dnVecDescr;

    dense_vec() = default;

    dense_vec(thrust::device_vector<float> &vec) {
        cusparseCreateDnVec(&dnVecDescr,
                            static_cast<int64_t>(vec.size()),
                            static_cast<void *>(thrust::raw_pointer_cast(vec.data())),
                            CUDA_R_32F);

        std::cout << "size = " << static_cast<int64_t>(vec.size()) << std::endl;
    }


};

struct csr_rect {
    cusparseSpMatDescr_t spMatDescr;

    int rows, cols;
    int nnz;

    thrust::device_vector<float> d_val_vec;
    thrust::device_vector<int> d_row_ptr_vec;
    thrust::device_vector<int> d_col_idx_vec;

    thrust::device_ptr<float> d_val;
    thrust::device_ptr<int> d_row_ptr;
    thrust::device_ptr<int> d_col_idx;

    csr_rect() = default;

    csr_rect(const eg::SparseMatrix<float, eg::RowMajor> &mat) {

        // metadata conversion
        rows = mat.rows();
        cols = mat.cols();
        nnz = mat.nonZeros();

        const int *h_row_ptr = mat.outerIndexPtr();
        const int *h_col_idx = mat.innerIndexPtr();
        const float *h_val = mat.valuePtr();
        int row_bytes = sizeof(int) * (rows + 1);
        int col_bytes = sizeof(int) * nnz;
        int val_bytes = sizeof(float) * nnz;

        // device init
        d_row_ptr_vec = thrust::device_vector<int>(rows + 1);
        d_col_idx_vec = thrust::device_vector<int>(nnz);
        d_val_vec = thrust::device_vector<float>(nnz);
        d_row_ptr = d_row_ptr_vec.data();
        d_col_idx = d_col_idx_vec.data();
        d_val = d_val_vec.data();

        // copy to device
        cudaMemcpy(D_row_ptr(), h_row_ptr, row_bytes, cudaMemcpyDefault);
        cudaMemcpy(D_col_idx(), h_col_idx, col_bytes, cudaMemcpyDefault);
        cudaMemcpy(D_val(), h_val, val_bytes, cudaMemcpyDefault);

        // create cusparse matrix descriptor
        cusparseCreateCsr(&spMatDescr,
                          rows,
                          cols,
                          nnz,
                          static_cast<void *>(D_row_ptr()),
                          static_cast<void *>(D_col_idx()),
                          static_cast<void *>(D_val()),
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_32F);


    }

    [[nodiscard]] float *D_val() const {
        return thrust::raw_pointer_cast(d_val);
    }

    [[nodiscard]] int *D_row_ptr() const {
        return thrust::raw_pointer_cast(d_row_ptr);
    }

    [[nodiscard]] int *D_col_idx() const {
        return thrust::raw_pointer_cast(d_col_idx);
    }

    void Axpby(const cusparseHandle_t &spHandle,
               float alpha, const dense_vec &x,
               float beta, const dense_vec &y) {

        // y = alpha * A * x + beta * y

        size_t bufferSize;

        void *d_mvBuffer = nullptr;


        cusparseSpMV_bufferSize(spHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                static_cast<const void *>(&alpha),
                                spMatDescr,
                                x.dnVecDescr,
                                static_cast<const void *>(&beta),
                                y.dnVecDescr,
                                CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                &bufferSize);

        cudaMalloc(&d_mvBuffer, bufferSize);

        cusparseSpMV(spHandle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     static_cast<void *>(&alpha),
                     spMatDescr,
                     x.dnVecDescr,
                     static_cast<void *>(&beta),
                     y.dnVecDescr,
                     CUDA_R_32F,
                     CUSPARSE_SPMV_ALG_DEFAULT,
                     d_mvBuffer);

        cudaDeviceSynchronize();

        cudaFree(d_mvBuffer);

    }

};

struct csr_matrix {


    thrust::device_vector<float> d_val_vec;
    thrust::device_vector<int> d_row_ptr_vec;
    thrust::device_vector<int> d_col_idx_vec;

    thrust::device_ptr<float> d_val;
    thrust::device_ptr<int> d_row_ptr;
    thrust::device_ptr<int> d_col_idx;

public:
    int n;
    int nnz;
    cusparseMatDescr_t descr;

    csr_matrix() = default;

    csr_matrix(const eg::SparseMatrix<float, eg::RowMajor> &mat) {

        if (not mat.IsRowMajor) {
            std::cerr << "Fatal: Incompatible matrix type! Row major (CSR format) expected." << std::endl;
            exit(1);
        }


        // init
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        // metadata conversion
        n = mat.rows();
        nnz = mat.nonZeros();

        // data conversion
        const int *h_row_ptr = mat.outerIndexPtr();
        const int *h_col_idx = mat.innerIndexPtr();
        const float *h_val = mat.valuePtr();
        int row_bytes = sizeof(int) * (n + 1);
        int col_bytes = sizeof(int) * nnz;
        int val_bytes = sizeof(float) * nnz;
        // 1. init device memory
        d_row_ptr_vec = thrust::device_vector<int>(n + 1);
        d_col_idx_vec = thrust::device_vector<int>(nnz);
        d_val_vec = thrust::device_vector<float>(nnz);
        d_row_ptr = d_row_ptr_vec.data();
        d_col_idx = d_col_idx_vec.data();
        d_val = d_val_vec.data();

        // 2. copy
        cudaMemcpy(D_row_ptr(), h_row_ptr, row_bytes, cudaMemcpyDefault);
        cudaMemcpy(D_col_idx(), h_col_idx, col_bytes, cudaMemcpyDefault);
        cudaMemcpy(D_val(), h_val, val_bytes, cudaMemcpyDefault);


    }


    [[nodiscard]] float *D_val() const {
        return thrust::raw_pointer_cast(d_val);
    }

    [[nodiscard]] int *D_row_ptr() const {
        return thrust::raw_pointer_cast(d_row_ptr);
    }

    [[nodiscard]] int *D_col_idx() const {
        return thrust::raw_pointer_cast(d_col_idx);
    }


};


class D_Cholesky {

    csr_matrix &csrMatrix;
    float tol;

public:

    explicit D_Cholesky(csr_matrix &_csrMatrix, float _tol = 1e-6) : csrMatrix(_csrMatrix), tol(_tol) {
    }

    // returns singularity
    int Solve(cusolverSpHandle_t solveHandle, const float *b, float *x) {
        int singularity;
        cusolverSpScsrlsvchol(solveHandle,
                              csrMatrix.n,
                              csrMatrix.nnz,
                              csrMatrix.descr,
                              csrMatrix.D_val(),
                              csrMatrix.D_row_ptr(),
                              csrMatrix.D_col_idx(),
                              b,
                              tol,
                              0,
                              x,
                              &singularity);

        cudaDeviceSynchronize();

        return singularity;
    }

};  