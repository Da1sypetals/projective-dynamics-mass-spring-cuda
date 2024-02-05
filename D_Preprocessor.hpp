#pragma once

#include "D_Solver.cuh"
#include <memory>


class D_Preprocessor {
    
    
public:

    std::shared_ptr<D_Solver> dSolver;

    D_Preprocessor(std::shared_ptr<D_Solver> _dSolver) : dSolver(_dSolver) {};


    void Init_M() {
        std::vector<eg::Triplet<float>> MTriplets;
        for (int ivertex = 0; ivertex < dSolver->cloth->numVertex; ivertex++) {
            for (int j = 0; j < 3; j++) {
                MTriplets.push_back(eg::Triplet<float>(3 * ivertex + j, 3 * ivertex + j, dSolver->cloth->mass));
            }
        }
        dSolver->h_M.resize(3 * dSolver->cloth->numVertex, 3 * dSolver->cloth->numVertex);
        dSolver->h_M.setFromTriplets(MTriplets.begin(), MTriplets.end());
    }

    void Init_L() {
        std::vector<eg::Triplet<float>> LTriplets;
        for (auto &&con: dSolver->cloth->constraints) {
            for (int j = 0; j < 3; j++) {
                LTriplets.push_back(eg::Triplet<float>(3 * con.istart + j, 3 * con.istart + j, dSolver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(3 * con.istart + j, 3 * con.iend + j, -dSolver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(3 * con.iend + j, 3 * con.iend + j, dSolver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(3 * con.iend + j, 3 * con.istart + j, -dSolver->cloth->k));
            }
        }
        dSolver->h_L.resize(3 * dSolver->cloth->numVertex, 3 * dSolver->cloth->numVertex);
        dSolver->h_L.setFromTriplets(LTriplets.begin(), LTriplets.end());
    }

    void Init_J() {
        std::vector<eg::Triplet<float>> JTriplets;
        for (int iconstraint = 0; iconstraint < dSolver->cloth->numConstraint; iconstraint++) {

            auto &con = dSolver->cloth->constraints[iconstraint];

            for (int j = 0; j < 3; j++) {
                JTriplets.push_back(eg::Triplet<float>(3 * con.istart + j, 3 * iconstraint + j, -dSolver->cloth->k));
                JTriplets.push_back(eg::Triplet<float>(3 * con.iend + j, 3 * iconstraint + j, dSolver->cloth->k));
            }
        }
        dSolver->h_J.resize(3 * dSolver->cloth->numVertex, 3 * dSolver->cloth->numConstraint);
        dSolver->h_J.setFromTriplets(JTriplets.begin(), JTriplets.end());
    }

    void D_CreateVec() {
        dSolver->d_d.resize(3 * dSolver->cloth->numConstraint);
        dSolver->d_x_prev.resize(3 * dSolver->cloth->numVertex);
        dSolver->d_x.resize(3 * dSolver->cloth->numVertex);
        dSolver->d_y.resize(3 * dSolver->cloth->numVertex);
        dSolver->d_b.resize(3 * dSolver->cloth->numVertex);
        dSolver->d_f_external.resize(3 * dSolver->cloth->numVertex);

    }

    void D_InitVec() {
        for (int irow = 0; irow < dSolver->cloth->nside; irow++) {
            for (int icol = 0; icol < dSolver->cloth->nside; icol++) {

                int idx = dSolver->index(irow, icol);

                // init pos {
                dSolver->d_x[3 * idx] =
                        dSolver->cloth->size * static_cast<float>(irow) / static_cast<float>(dSolver->cloth->nside - 1);
                dSolver->d_x[3 * idx + 1] =
                        dSolver->cloth->size * static_cast<float>(icol) / static_cast<float>(dSolver->cloth->nside - 1);
                dSolver->d_x[3 * idx + 2] = 0;
                dSolver->d_x_prev[3 * idx] = dSolver->d_x[3 * idx];
                dSolver->d_x_prev[3 * idx + 1] = dSolver->d_x[3 * idx + 1];
                dSolver->d_x_prev[3 * idx + 2] = dSolver->d_x[3 * idx + 2];
                // }

                // init force {
                dSolver->d_f_external[3 * idx] = 0;
                dSolver->d_f_external[3 * idx + 1] = 0;
                dSolver->d_f_external[3 * idx + 2] = gravity;
                // }


            }
        }

    }

    void H_CreateVec() {

        dSolver->h_x.resize(3 * dSolver->cloth->numVertex);

    }

    void InitDevDenseVec() {
        dSolver->dn_d = dense_vec(dSolver->d_d);
        dSolver->dn_b = dense_vec(dSolver->d_b);
    }

    void InitDevMat() {
        dSolver->d_Y = csr_matrix(dSolver->h_Y);
        dSolver->d_M = csr_matrix(dSolver->h_M);
        dSolver->d_L = csr_matrix(dSolver->h_L);

        dSolver->d_J = csr_rect(dSolver->h_J);


    }

    void InitCholesky() {
        dSolver->cholesky = std::make_unique<D_Cholesky>(dSolver->d_Y, 1e-6);

    }

    void InitDevConstraints() {
        thrust::copy(dSolver->cloth->constraints.begin(), dSolver->cloth->constraints.end(),
                     dSolver->d_constraints.begin());

    };


    void Init() {

        dSolver->cloth->InitConstraints();

        // init matrices at host
        Init_J();
        Init_L();
        Init_M();
        // initialize Y
        dSolver->h_Y = dSolver->h_M + dt2 * dSolver->h_L;

        // init vec at host
        H_CreateVec();

        // device init
        D_CreateVec();
        D_InitVec();

        // After device init, create device dense vector descriptor
        InitDevDenseVec();

        // move matrices to device
        InitDevMat();

        // init cholesky
        InitCholesky();

        // init device constraints
        InitDevConstraints();


        dSolver->initialized = true;

    }


};



