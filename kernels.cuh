#include <device_launch_parameters.h>

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

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
    float3 dir_times_restLength = normalize(dir) * con_ptr->restLength;
//    printf("%d: [%.3f, %.3f, %.3f]\n", tid, dir_times_restLength.x, dir_times_restLength.y, dir_times_restLength.z);

    d_d[tid * 3 + 0] = dir_times_restLength.x;
    d_d[tid * 3 + 1] = dir_times_restLength.y;
    d_d[tid * 3 + 2] = dir_times_restLength.z;

    printf("%d: [%.3f, %.3f, %.3f]\n",
           tid,
           static_cast<double>(d_d[tid * 3 + 0]),
           static_cast<double>(d_d[tid * 3 + 1]),
           static_cast<double>(d_d[tid * 3 + 2]));


}


