#include <vector>
#include "Types.hpp"

#pragma once

class Cloth {

public:
    int nside;
    int numVertex;
    int numConstraint;
    float mass;
    float restLength1, restLength2;
    float size; // area = size * size
    float k; // stiffness

    std::vector<Constraint> constraints;

    Cloth(int _nside, float _size, float _k) : nside(_nside), size(_size), k(_k) {
        numVertex = nside * nside;
        mass = 1;
        restLength1 = size / (nside - 1);
        restLength2 = std::sqrt(2) * restLength1;

    }

    int index(int irow, int icol) {
        return irow * nside + icol;
    }

    void InitConstraints() {

        for (int irow = 0; irow < nside; irow++) {
            for (int icol = 0; icol < nside; icol++) {
                if (icol + 1 < nside) {
                    constraints.push_back(Constraint(index(irow, icol), index(irow, icol + 1), restLength1));
                }
                if (irow + 1 < nside) {
                    constraints.push_back(Constraint(index(irow, icol), index(irow + 1, icol), restLength1));
                }
                if (irow + 1 < nside and icol + 1 < nside) {
                    constraints.push_back(Constraint(index(irow, icol), index(irow + 1, icol + 1), restLength2));
                }
                if (irow + 1 < nside and icol - 1 >= 0) {
                    constraints.push_back(Constraint(index(irow, icol), index(irow + 1, icol - 1), restLength2));
                }

            }
        }

        numConstraint = constraints.size();

    }


};