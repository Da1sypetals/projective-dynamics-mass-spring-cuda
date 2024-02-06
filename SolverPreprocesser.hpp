#pragma once

#include <utility>
#include <vector>
#include <memory>
#include <Eigen/Sparse>
#include "Types.hpp"
#include "Config.hpp"
#include "Cloth.hpp"
#include "H_Solver.hpp"


class SolverPreprocessor {

public:


    std::shared_ptr<H_Solver> hSolver;

    SolverPreprocessor(std::shared_ptr<H_Solver> _hSolver) : hSolver(_hSolver) {};


    void Init_M() {
        std::vector<eg::Triplet<float>> MTriplets;
        for (int ivertex = 0; ivertex < hSolver->cloth->numVertex; ivertex++) {
            for (int j = 0; j < 3; j++) {
                MTriplets.push_back(eg::Triplet<float>(3 * ivertex + j, 3 * ivertex + j, hSolver->cloth->mass));
            }
        }
        hSolver->h_M.resize(3 * hSolver->cloth->numVertex, 3 * hSolver->cloth->numVertex);
        hSolver->h_M.setFromTriplets(MTriplets.begin(), MTriplets.end());
    }

    void Init_L() {
        std::vector<eg::Triplet<float>> LTriplets;
        for (auto &&con: hSolver->cloth->constraints) {
            for (int j = 0; j < 3; j++) {
                LTriplets.push_back(eg::Triplet<float>(3 * con.istart + j, 3 * con.istart + j, hSolver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(3 * con.istart + j, 3 * con.iend + j, -hSolver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(3 * con.iend + j, 3 * con.iend + j, hSolver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(3 * con.iend + j, 3 * con.istart + j, -hSolver->cloth->k));
            }
        }
        hSolver->h_L.resize(3 * hSolver->cloth->numVertex, 3 * hSolver->cloth->numVertex);
        hSolver->h_L.setFromTriplets(LTriplets.begin(), LTriplets.end());
    }

    void Init_J() {
        std::vector<eg::Triplet<float>> JTriplets;
        for (int iconstraint = 0; iconstraint < hSolver->cloth->numConstraint; iconstraint++) {

            auto &con = hSolver->cloth->constraints[iconstraint];

            for (int j = 0; j < 3; j++) {
                JTriplets.push_back(eg::Triplet<float>(3 * con.istart + j, 3 * iconstraint + j, -hSolver->cloth->k));
                JTriplets.push_back(eg::Triplet<float>(3 * con.iend + j, 3 * iconstraint + j, hSolver->cloth->k));
            }
        }
        hSolver->h_J.resize(3 * hSolver->cloth->numVertex, 3 * hSolver->cloth->numConstraint);
        hSolver->h_J.setFromTriplets(JTriplets.begin(), JTriplets.end());
    }

    void CreateVec() {
        hSolver->d = eg::VectorXf(3 * hSolver->cloth->numConstraint);
        hSolver->x_prev = eg::VectorXf(3 * hSolver->cloth->numVertex);
        hSolver->x = eg::VectorXf(3 * hSolver->cloth->numVertex);
        hSolver->y = eg::VectorXf(3 * hSolver->cloth->numVertex);
        hSolver->b = eg::VectorXf(3 * hSolver->cloth->numVertex);
        hSolver->f_external = eg::VectorXf(3 * hSolver->cloth->numVertex);

    }

    void InitVec() {
        for (int irow = 0; irow < hSolver->cloth->nside; irow++) {
            for (int icol = 0; icol < hSolver->cloth->nside; icol++) {

                int idx = hSolver->index(irow, icol);

                // init pos {
                hSolver->x[3 * idx] =
                        hSolver->cloth->size * static_cast<float>(irow) / static_cast<float>(hSolver->cloth->nside - 1);
                hSolver->x[3 * idx + 1] = 0;

                hSolver->x[3 * idx + 2] = hSolver->cloth->size * static_cast<float>(icol) / static_cast<float>(hSolver->cloth->nside - 1);
                hSolver->x_prev[3 * idx] = hSolver->x[3 * idx];
                hSolver->x_prev[3 * idx + 1] = hSolver->x[3 * idx + 1];
                hSolver->x_prev[3 * idx + 2] = hSolver->x[3 * idx + 2];
                // }

                // init force {
                hSolver->f_external[3 * idx] = 0;
                hSolver->f_external[3 * idx + 1] = gravity;
                hSolver->f_external[3 * idx + 2] = 0;
                // }


            }
        }

    }

    void Init() {

        hSolver->cloth->InitConstraints();

        CreateVec();
        InitVec();

        Init_J();
        Init_L();
        Init_M();

        hSolver->h_Y = hSolver->h_M + dt2 * hSolver->h_L;

//        eg::SelfAdjointEigenSolver<eg::SparseMatrix<float>> eigSolver;
//        eigSolver.compute(hSolver->h_Y, eg::EigenvaluesOnly);
//        eg::VectorXf eig = eigSolver.eigenvalues();
//        float prod = 1;
//        for (int i = 0; i < eig.size(); ++i) {
//            std::cout << eig[i] << " ";
//            prod *= eig[i] > 0 ? 1 : -1;
//        }
//        std::cout << std::endl << std::endl << "prod = " << prod;
//        system("pause");


        hSolver->llt.compute(hSolver->h_Y);
        if (hSolver->llt.info() != eg::Success) {
            std::cerr << "Fatal: LLT decomposition failed! code " << hSolver->llt.info() << std::endl;
            system("pause");
            exit(1);
        }

        hSolver->initialized = true;

    }


};