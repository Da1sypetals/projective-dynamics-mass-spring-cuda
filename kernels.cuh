#include "helper_math.h"
#include <device_launch_parameters.h>

__global__ void D_LocalStep(int numConstraint,
                            thrust::device_ptr<Constraint> d_constraints,
                            thrust::device_ptr<float> d_x,
                            thrust::device_ptr<float> d_d) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numConstraint) {
        return;
    }

    thrust::device_ptr<Constraint> con_ptr = d_constraints + tid;

    float d0 = d_x[con_ptr->iend * 3 + 0] - d_x[con_ptr->istart * 3 + 0];
    float d1 = d_x[con_ptr->iend * 3 + 1] - d_x[con_ptr->istart * 3 + 1];
    float d2 = d_x[con_ptr->iend * 3 + 2] - d_x[con_ptr->istart * 3 + 2];

    float3 dir = {d0, d1, d2};
    dir = normalize(dir);

    d_d[tid * 3 + 0] = dir.x * con_ptr->restLength;
    d_d[tid * 3 + 1] = dir.y * con_ptr->restLength;
    d_d[tid * 3 + 2] = dir.z * con_ptr->restLength;

}


