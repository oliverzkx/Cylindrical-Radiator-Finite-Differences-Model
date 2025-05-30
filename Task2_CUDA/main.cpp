#include <iostream>
#include <vector>
#include <getopt.h>
#include "heat_utils.h"

// CUDA kernel interface (in heat_kernel.cu)
extern "C" void launch_cuda_heat(float* host_prev, int n, int m, int p, bool use_stop, float stop_avg, bool show_timing);

#define DEFAULT_N 64
#define DEFAULT_M 64
#define DEFAULT_P 100

int main(int argc, char* argv[]) {
    int n = DEFAULT_N, m = DEFAULT_M, p = DEFAULT_P;
    float stop_avg = 0.0f;
    bool use_stop_flag = false;  // Command-line flag for stopping
    bool use_cpu = true;         // Run CPU version unless -c disables it
    bool show_timing = false;

    // Parse command-line options
    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:a:ct")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': use_stop_flag = true; stop_avg = atof(optarg); break;
            case 'c': use_cpu = false; break;  // -c disables CPU
            case 't': show_timing = true; break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-n rows] [-m cols] [-p iterations] [-a avg_stop] [-c] [-t]\n";
                return 1;
        }
    }

    // Initialize matrix with initial values
    std::vector<float> data(n * m, 0.0f);
    for (int i = 0; i < n; ++i) {
        data[i * m + 0] = 0.98f * (i + 1) * (i + 1) / (float)(n * n);  // boundary
        for (int j = 1; j < m; ++j)
            data[i * m + j] = ((float)(m - j) * (m - j)) / (float)(m * m);  // interior
    }

    // CPU & GPU data
    std::vector<float> cpu_data = data;
    std::vector<float> gpu_data = data;
    std::vector<float> cpu_avg(n, 0.0f), gpu_avg(n, 0.0f);

    // 🔧 Correct flag usage for stopping
    bool use_stop_cpu = false;
    bool use_stop_gpu = use_stop_flag;

    // Run CPU
    if (use_cpu) {
        std::cout << "🔁 Running CPU simulation...\n";
        cpu_heat_propagation(cpu_data.data(), n, m, p, use_stop_cpu, stop_avg, false);
    }

    // Run GPU
    std::cout << "🔁 Running GPU simulation...\n";
    launch_cuda_heat(gpu_data.data(), n, m, p, use_stop_gpu, stop_avg, show_timing);

    // Print top-left matrix values
    std::cout << "First 3 values of each row:\n";
    for (int i = 0; i < std::min(n, 5); ++i) {
        for (int j = 0; j < std::min(m, 3); ++j)
            std::cout << gpu_data[i * m + j] << " ";
        std::cout << "\n";
    }

    // Compare CPU and GPU
    if (use_cpu) {
        std::cout << "🔍 Comparing CPU and GPU results after " << p << " steps...\n";
        compare_results(cpu_data.data(), gpu_data.data(), n * m,
                        cpu_avg.data(), gpu_avg.data(), n);
        print_compare_debug(cpu_data.data(), gpu_data.data(), n, m);
    }

    return 0;
}
