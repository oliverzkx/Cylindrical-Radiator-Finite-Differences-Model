#include "heat_utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std::chrono;

void cpu_heat_propagation(float* data, int n, int m, int p,
                          bool use_stop, float stop_avg, bool show_timing) {
    std::vector<float> prev(data, data + n * m);
    std::vector<float> next(n * m, 0.0f);

    auto start = high_resolution_clock::now();

    for (int step = 0; step < p; ++step) {
        for (int u = 1; u < n - 1; ++u) {
            for (int j = 2; j < m - 2; ++j) {
                next[u * m + j] =
                    (1.60f * prev[u * m + j - 2] +
                     1.55f * prev[u * m + j - 1] +
                     0.60f * prev[u * m + j + 1] +
                     0.25f * prev[u * m + j + 2] +
                     1.00f * prev[u * m + j]) / 5.0f;
            }
        }

        // Preserve boundary rows
        for (int j = 0; j < m; ++j) {
            next[0 * m + j] = prev[0 * m + j];             // top row
            next[(n - 1) * m + j] = prev[(n - 1) * m + j]; // bottom row
        }

        // Preserve boundary columns
        for (int u = 0; u < n; ++u) {
            next[u * m + 0] = prev[u * m + 0];
            next[u * m + 1] = prev[u * m + 1];
            next[u * m + m - 2] = prev[u * m + m - 2];
            next[u * m + m - 1] = prev[u * m + m - 1];
        }

        std::swap(prev, next);

        if (use_stop) {
            for (int i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < m; ++j)
                    sum += prev[i * m + j];
                float avg = sum / m;
                if (avg >= stop_avg) {
                    std::cout << "🔴 CPU stopped at iteration " << step + 1 << "\n";
                    break;
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    if (show_timing) {
        auto duration = duration_cast<milliseconds>(end - start).count();
        std::cout << "[CPU] Propagation Time: " << duration << " ms\n";
    }

    std::copy(prev.begin(), prev.end(), data);
}


void compare_results(const float* cpu_data, const float* gpu_data, int size,
                     const float* cpu_avg, const float* gpu_avg, int rows, float threshold) {
    float max_matrix_diff = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < size; ++i) {
        float diff = std::abs(cpu_data[i] - gpu_data[i]);
        if (diff > threshold) mismatches++;
        if (diff > max_matrix_diff) max_matrix_diff = diff;
    }

    std::cout << "[Compare] Max matrix diff: " << max_matrix_diff;
    if (max_matrix_diff > threshold)
        std::cout << " ❌ mismatch found!\n";
    else
        std::cout << " ✅\n";

    if (cpu_avg && gpu_avg) {
        float max_avg_diff = 0.0f;
        for (int i = 0; i < rows; ++i) {
            float diff = std::abs(cpu_avg[i] - gpu_avg[i]);
            if (diff > max_avg_diff) max_avg_diff = diff;
        }

        std::cout << "[Compare] Max avg diff: " << max_avg_diff;
        if (max_avg_diff > threshold)
            std::cout << " ❌ mismatch found!\n";
        else
            std::cout << " ✅\n";
    }
}

void compare_results_verbose(const float* cpu, const float* gpu, int size, float threshold) {
    float max_diff = 0.0f;
    int mismatch_count = 0;

    for (int i = 0; i < size; ++i) {
        float diff = std::abs(cpu[i] - gpu[i]);
        if (diff > threshold) {
            std::cout << "[Mismatch] Index: " << i
                      << " | CPU: " << cpu[i]
                      << " | GPU: " << gpu[i]
                      << " | Diff: " << diff << "\n";
            mismatch_count++;
        }
        if (diff > max_diff)
            max_diff = diff;
    }

    if (mismatch_count > 0) {
        std::cout << "[Compare] 🔺 Total mismatches above " << threshold << ": " << mismatch_count << "\n";
    } else {
        std::cout << "[Compare] ✅ All values within threshold.\n";
    }

    std::cout << "[Compare] Max absolute difference: " << max_diff << "\n";
}
