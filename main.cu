#include <iostream>
#include "raylib.h"
#include "SolverPreprocesser.hpp"
#include "Types.hpp"
#include "Window3d.hpp"
#include "Timer.hpp"
#include "Cloth.hpp"
#include "Config.hpp"

#include "D_Solver.cuh"
#include "D_Preprocessor.hpp"
#include "H_Solver.hpp"
#include "SolverPreprocesser.hpp"
//
//
//int main() {
//
//    int n;
//    float size;
//    float k;
//    int numSubstep;
//    int n_iter = 10;
//    bool log_time;
//
//    std::cout << "number of each side, cloth size, stiffness k, number of substep, log time or not" << std::endl;
//    std::cin >> n >> size >> k >> numSubstep >> log_time;
//
//    // init handles {
//    cusolverSpHandle_t cusolverSpHandle;
//    cusolverSpCreate(&cusolverSpHandle);
//    cusparseHandle_t cusparseHandle;
//    cusparseCreate(&cusparseHandle);
//
//    // }
//
//
//    // init device solver {
//
//
//    std::shared_ptr<Cloth> cloth = std::make_shared<Cloth>(n, size, k);
//
//    std::shared_ptr<D_Solver> dSolver = std::make_shared<D_Solver>(cloth, n_iter);
//    dSolver->SetHandles(cusolverSpHandle, cusparseHandle);
//
//    std::shared_ptr<D_Preprocessor> pre = std::make_shared<D_Preprocessor>(dSolver);
//    pre->Init();
//
//    std::cout << ">>> Preprocessing done...\n" << std::endl;
//
//    dSolver->AddFixed(0, 0);
//    dSolver->AddFixed(0, n - 1);
//
//    std::cout << ">>> Iteration per substep: " << n_iter << std::endl << std::endl;
//
//    // test 3
////    std::cout << "row ptr:" << std::endl;
////    for (int i = 0; i < dSolver->h_M.outerSize() + 1; i++) {
////        std::cout << dSolver->d_M.d_row_ptr_vec[i] << std::endl;
////    }
////    std::cout << "\n\ncol idx:" << std::endl;
////    for (int i = 0; i < dSolver->h_M.nonZeros(); i++) {
////        std::cout << dSolver->d_M.d_col_idx_vec[i] << std::endl;
////    }
////    std::cout << "\n\nval:" << std::endl;
////    for (int i = 0; i < dSolver->h_M.nonZeros(); i++) {
////        std::cout << dSolver->d_M.d_val_vec[i] << std::endl;
////    }
//
//
//
//    // test 2
////    std::cout << "row ptr:" << std::endl;
////    for (int i = 0; i < dSolver->h_M.outerSize() + 1; i++) {
////        std::cout << dSolver->h_M.outerIndexPtr()[i] << std::endl;
////    }
////
////    std::cout << "\n\ncol idx:" << std::endl;
////    for (int i = 0; i < dSolver->h_M.nonZeros(); i++) {
////        std::cout << dSolver->h_M.innerIndexPtr()[i] << std::endl;
////    }
////
////    std::cout << "\n\nval:" << std::endl;
////    for (int i = 0; i < dSolver->h_M.nonZeros(); i++) {
////        std::cout << dSolver->h_M.valuePtr()[i] << std::endl;
////    }
//
//
//    bool GUI;
//    std::cout << "enable GUI?" << std::endl;
//    std::cin >> GUI;
//
//
//    // }
//
////     def update {
//
//    auto update = [&] {
//
//        for (int substep = 0; substep < numSubstep; substep++) {
//
////            std::cout << "substep :" << substep << std::endl;
//            dSolver->Step();
//
//        }
//    };
//
//    // }
//
//    Timer updateTimer, drawTimer;
//
//    if (GUI) {
////        const int screenWidth = 1200;
////        const int screenHeight = 1200;
////        const float radius = size / static_cast<float>(n) * .15f;
////
////        InitWindow(screenWidth, screenHeight, "Projective dynamimcs");
////        SetTargetFPS(60);
////        Window3d window3d;
////        window3d.Init();
////        while (not WindowShouldClose()) {
////
////
////            updateTimer.start();
////            if (!window3d.pause) {
////                update();
////
////            }
////            updateTimer.stop();
//////        std::cout << ">>> update" << std::endl;
////
////
////            BeginDrawing();
////            ClearBackground(RAYWHITE);
////
////            window3d.Update();
////            window3d.Begin();
////
////            drawTimer.start();
////            for (int i = 0; i < hSolver->cloth->numVertex; i++) {
////
////                Vector3 center = {hSolver->x[3 * i], hSolver->x[3 * i + 1], hSolver->x[3 * i + 2]};
////                DrawPoint3D(center, RED);
//////            DrawSphere(center, radius, RED);
////
////            }
////            drawTimer.stop();
////
////
////            window3d.End();
////
////
////            EndDrawing();
////
////            if (log_time) {
////                printf("update time: %f, draw time: %f\n\n", updateTimer.getTime(), drawTimer.getTime());
////            }
////
////
////        }
////
////        CloseWindow();
//    } else {
//        while (true) {
//
//
//            updateTimer.start();
//            update();
//            updateTimer.stop();
//            std::cout << ">>> update" << std::endl;
//
//
//            for (int irow = 0; irow < dSolver->cloth->nside; irow++) {
//                for (int icol = 0; icol < dSolver->cloth->nside; icol++) {
//                    printf("[%.3f, %.3f, %.3f]  ",
//                           dSolver->h_x[3 * dSolver->index(irow, icol)],
//                           dSolver->h_x[3 * dSolver->index(irow, icol) + 1],
//                           dSolver->h_x[3 * dSolver->index(irow, icol) + 2]);
//                }
//                std::cout << std::endl;
//            }
//
//            std::cout << "\n --- Press Enter to continue --- \n";
//            std::cin.get();
//
//
//        }
//    }
//
//    cusparseDestroy(cusparseHandle);
//    cusolverSpDestroy(cusolverSpHandle);
//
//
//    return 0;
//
//}
//
//


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <cusparse.h>
#include "MatrixTypes.hpp"
#include <Eigen/Sparse>
#include "Types.hpp"

int main() {
    // Initialize cuSPARSE
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);

    eg::SparseMatrix<float, eg::RowMajor> h_mat;

    std::vector<eg::Triplet<float>> tr;
    tr.push_back(eg::Triplet<float>(0, 0, 1));
    tr.push_back(eg::Triplet<float>(0, 1, 3));
    tr.push_back(eg::Triplet<float>(0, 2, 4));
    tr.push_back(eg::Triplet<float>(1, 0, 0));
    tr.push_back(eg::Triplet<float>(1, 1, 2));
    tr.push_back(eg::Triplet<float>(1, 2, 6));
    tr.push_back(eg::Triplet<float>(3, 0, 3));
    tr.push_back(eg::Triplet<float>(3, 1, 6));
    tr.push_back(eg::Triplet<float>(3, 2, 2));
    h_mat.setFromTriplets(tr.begin(), tr.end());

    csr_rect d_mat(h_mat);

    // Initialize your dense 3-vectors on the device
    thrust::device_vector<float> d_x_vec = {7, 8, 9};  // Initialize vector x
    dense_vec d_x(d_x_vec);
    thrust::device_vector<float> d_y_vec = {1, 3, 1};  // Initialize vector y
    dense_vec d_y(d_y_vec);

    // Define scalar constants for the operation
    float alpha = 1.0f;
    float beta = 1.5f;

    // Perform Axpby operation
    d_mat.Axpby(handle, alpha, d_x, beta, d_y);

    // Transfer the result from the device to the host
    thrust::host_vector<float> h_y_vec(d_y_vec.size());
    thrust::copy(d_y_vec.begin(), d_y_vec.end(), h_y_vec.begin());

    // Print out the results
    std::cout << "The resulting vector is: [ ";
    for (const auto &val: h_y_vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    // Destroy cuSPARSE handle
    cusparseDestroy(handle);

    return 0;
}

