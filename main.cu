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


int main() {

    int n;
    float size;
    float k;
    int numSubstep;
    int n_iter = 10;
    bool log_time;

    std::cout << "number of each side, cloth size, stiffness k, number of substep, log time or not" << std::endl;
    std::cin >> n >> size >> k >> numSubstep >> log_time;

    // init handles {
    cusolverSpHandle_t cusolverSpHandle;
    cusolverSpCreate(&cusolverSpHandle);
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);

    // }

    // init solver {


    std::shared_ptr<Cloth> cloth = std::make_shared<Cloth>(n, size, k);

    std::shared_ptr<D_Solver> dSolver = std::make_shared<D_Solver>(cloth, n_iter);
    dSolver->SetHandles(cusolverSpHandle, cusparseHandle);

    std::shared_ptr<D_Preprocessor> pre = std::make_shared<D_Preprocessor>(dSolver);
    pre->Init();

    std::cout << ">>> Preprocessing done...\n" << std::endl;

    dSolver->AddFixed(0, 0);
    dSolver->AddFixed(0, n - 1);

    std::cout << ">>> Iteration per substep: " << n_iter << std::endl << std::endl;

    for (int i = 0; i < cloth->numVertex; i++) {
        std::cout << dSolver->d_x[3 * i] << " ";
        std::cout << dSolver->d_x[3 * i + 1] << " ";
        std::cout << dSolver->d_x[3 * i + 2] << " ";
        std::cout << std::endl;
    }



    // }

    // def update {

//    auto update = [&] {
//
//        for (int substep = 0; substep < numSubstep; substep++) {
//
////            std::cout << "substep :" << substep << std::endl;
//            hSolver->Step();
//
//        }
//    };
//
//    // }
//
//    Timer updateTimer, drawTimer;
//
//    const int screenWidth = 1200;
//    const int screenHeight = 1200;
//    const float radius = size / static_cast<float>(n) * .15f;
//
//    InitWindow(screenWidth, screenHeight, "Projective dynamimcs");
//    SetTargetFPS(60);
//    Window3d window3d;
//    window3d.Init();
//    while (not WindowShouldClose()) {
//
//        updateTimer.start();
//        if (!window3d.pause) {
//            update();
//
//        }
//        updateTimer.stop();
////        std::cout << ">>> update" << std::endl;
//
//
//        BeginDrawing();
//        ClearBackground(RAYWHITE);
//
//        window3d.Update();
//        window3d.Begin();
//
//        drawTimer.start();
//        for (int i = 0; i < hSolver->cloth->numVertex; i++) {
//
//            Vector3 center = {hSolver->x[3 * i], hSolver->x[3 * i + 1], hSolver->x[3 * i + 2]};
//            DrawPoint3D(center, RED);
////            DrawSphere(center, radius, RED);
//
//        }
//        drawTimer.stop();
//
//
//        window3d.End();
//
//
//        EndDrawing();
//
//        if (log_time) {
//            printf("update time: %f, draw time: %f\n\n", updateTimer.getTime(), drawTimer.getTime());
//        }
//
//
//    }
//
//    CloseWindow();
    return 0;

}


