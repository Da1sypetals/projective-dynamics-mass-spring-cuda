#include <chrono>

class Timer {
public:
    Timer() : time(0.0f), started(false) {}

    void start() {
        if (!started) {
            started = true;
            startTime = std::chrono::high_resolution_clock::now();
        }
    }

    void stop() {
        if (started) {
            started = false;
            auto endTime = std::chrono::high_resolution_clock::now();
            auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime);
            time += timeSpan.count();
        }
    }

    void reset() {
        started = false;
        time = 0.0f;
    }

    float getTime() const {
        return time;
    }

private:
    bool started;
    float time;
    std::chrono::high_resolution_clock::time_point startTime;
};