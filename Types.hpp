#pragma once

#include <Eigen/Eigen>

namespace eg = Eigen;


struct Constraint {
    int istart, iend;
    float restLength;

    Constraint(int _istart, int _iend, float _restLength) : istart(_istart), iend(_iend), restLength(_restLength) {}

};