#pragma once

#include <memory>
#include "SolverPreprocesser.hpp"

class H_Solver {

    friend class SolverPreprocessor;

private:
    bool initialized;

public:

    int n_iter;

    std::shared_ptr<Cloth> cloth;

    eg::SparseMatrix<float, eg::RowMajor> h_Y;
    eg::SparseMatrix<float, eg::RowMajor> h_M;
    eg::SparseMatrix<float, eg::RowMajor> h_L;
    eg::SparseMatrix<float, eg::RowMajor> h_J;

    eg::VectorXf d;
    eg::VectorXf x_prev;
    eg::VectorXf x;
    eg::VectorXf y;
    eg::VectorXf b;
    eg::VectorXf f_external;

    eg::SimplicialLLT<eg::SparseMatrix<float>> llt;

    std::vector<std::pair<int, eg::Vector3f>> fixed;


    // Utility {

    int index(int irow, int icol) {
        return irow * cloth->nside + icol;
    }

    // }

    // Inner {

    void LocalStep() {
        for (int iconstraint = 0; iconstraint < cloth->numConstraint; iconstraint++) {
//            std::cout << "        icons = " << iconstraint << std::endl;
            auto &con = cloth->constraints[iconstraint];

            float d0 = x[con.iend * 3 + 0] - x[con.istart * 3 + 0];
            float d1 = x[con.iend * 3 + 1] - x[con.istart * 3 + 1];
            float d2 = x[con.iend * 3 + 2] - x[con.istart * 3 + 2];

            eg::Vector3f dir;
            dir << d0, d1, d2;
            dir.normalize();

            d[iconstraint * 3 + 0] = dir[0] * con.restLength;
            d[iconstraint * 3 + 1] = dir[1] * con.restLength;
            d[iconstraint * 3 + 2] = dir[2] * con.restLength;
        }
    }

    void GlobalStep() {
        b = dt2 * h_J * d + y + dt2 * f_external;
        x = llt.solve(b);

    }

    // }


    // API {

    H_Solver(std::shared_ptr<Cloth> _cloth, int _n_iter) : cloth(_cloth), n_iter(_n_iter) {

        initialized = false;

    }


    void Step() {


//        y = ((2 - preservation) * x - (1 - preservation) * x_prev);
        y = h_M * ((2 - preservation) * x - (1 - preservation) * x_prev);
        x_prev = x;

        for (int iter = 0; iter < n_iter; iter++) {
            LocalStep();
//            std::cout << iter << "Local" << std::endl;
            GlobalStep();
//            std::cout << iter << "Global" << std::endl;

        }

//        x += dt2 * f_external;

        for (auto &&[ifixed, fixpos]: fixed) {
            x[3 * ifixed] = fixpos(0);
            x[3 * ifixed + 1] = fixpos(1);
            x[3 * ifixed + 2] = fixpos(2);
        }


    }


    void AddFixed(int irow, int icol) {

        if (not initialized) {
            std::cerr << "Fatal: Not initialized, cannot add fixed vertex!" << std::endl;
            exit(1);
        }

        int idx = index(irow, icol);
        fixed.push_back(std::make_pair(index(irow, icol),
                                       eg::Vector3f(x[3 * idx], x[3 * idx + 1], x[3 * idx + 2])));
    }


    // }


};