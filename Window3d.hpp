class Window3d {

public:

    Camera3D camera;
    bool pause;


    void Init() {
        camera = {0};
        camera.position = {10.0f, 10.0f, 10.0f}; // Camera position
        camera.target = {0.0f, 0.0f, 0.0f};      // Camera looking at point
        camera.up = {0.0f, 1.0f, 0.0f};          // Camera up vector (rotation towards target)
        camera.fovy = 45.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type


        DisableCursor();
        pause = false;
    }

    void Update() {
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            UpdateCamera(&camera, CAMERA_FREE);

        }

        if (IsKeyPressed('R')) {
            camera.target = {0.0f, 0.0f, 0.0f};
        }

        if (IsKeyPressed(' ')) {
            pause = not pause;
        }
    }

    void Begin() {
        BeginMode3D(camera);
        DrawGrid(10, 1.0f);
    }

    void End() {
        EndMode3D();
    }

};