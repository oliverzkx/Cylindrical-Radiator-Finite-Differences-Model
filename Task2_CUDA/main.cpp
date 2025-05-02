#include <iostream>
#include <vector>
#include <getopt.h>
#include "heat_utils.h"

// CUDA kernel interface (in heat_kernel.cu)
extern "C" void launch_cuda_heat(float* host_prev, int n, int m, int p, bool use_stop, float stop_avg);

#define DEFAULT_N 64
#define DEFAULT_M 64
#define DEFAULT_P 100

int main(int argc, char* argv[]) {
    int n = DEFAULT_N, m = DEFAULT_M, p = DEFAULT_P;
    float stop_avg = 0.0f;
    bool use_stop = false;
    bool use_cpu = false;
    bool show_timing = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:a:ct")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': use_stop = true; stop_avg = atof(optarg); break;
            case 'c': use_cpu = true; break;
            case 't': show_timing = true; break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-n rows] [-m cols] [-p iterations] [-a avg_stop] [-c] [-t]\n";
                return 1;
        }
    }

    std::vector<float> data(n * m, 0.0f);

    // Init matrix: col 0 with boundary, others with initial temp
    for (int i = 0; i < n; ++i) {
        data[i * m + 0] = 0.98f * (i + 1) * (i + 1) / (float)(n * n);
        for (int j = 1; j < m; ++j)
            data[i * m + j] = ((float)(m - j) * (m - j)) / (float)(m * m);
    }

    std::vector<float> cpu_data = data; // backup for CPU fallback
    std::vector<float> gpu_data = data;
    std::vector<float> cpu_avg(n, 0.0f), gpu_avg(n, 0.0f);

    if (use_cpu) {
        cpu_heat_propagation(cpu_data.data(), n, m, p, use_stop, stop_avg, show_timing);
    } else {
        launch_cuda_heat(gpu_data.data(), n, m, p, use_stop, stop_avg);
    }

    // Output matrix: first 3 values of each row
    std::cout << "First 3 values of each row:\n";
    for (int i = 0; i < std::min(n, 5); ++i) {
        for (int j = 0; j < std::min(m, 3); ++j) {
            std::cout << (use_cpu ? cpu_data[i * m + j] : gpu_data[i * m + j]) << " ";
        }
        std::cout << "\n";
    }

    // Optional compare CPU vs GPU
    if (show_timing && !use_cpu) {
        compare_results(cpu_data.data(), gpu_data.data(), n * m,
                        cpu_avg.data(), gpu_avg.data(), n);
    }

    return 0;
}
