#include <iostream>
#include <vector>
#include <getopt.h>
#include <chrono>  // For CPU timing
#include "heat_utils.h"

// CUDA kernel interface (defined in heat_kernel.cu)
extern "C" void launch_cuda_heat(double* host_prev, int n, int m, int p,
                                 bool use_stop, double stop_avg, bool show_timing);

#define DEFAULT_N 64
#define DEFAULT_M 64
#define DEFAULT_P 100

int main(int argc, char* argv[]) {
    int n = DEFAULT_N, m = DEFAULT_M, p = DEFAULT_P;
    double stop_avg = 0.0;
    bool use_stop_flag = false;
    bool use_cpu = true;         // Run CPU by default unless disabled with -c
    bool show_timing = false;    // Show GPU timing with -t

    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:a:ct")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': use_stop_flag = true; stop_avg = atof(optarg); break;
            case 'c': use_cpu = false; break;  // Disable CPU with -c
            case 't': show_timing = true; break; // Show timing with -t
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-n rows] [-m cols] [-p steps] [-a avg_stop] [-c] [-t]\n";
                return 1;
        }
    }

    // Initialize initial heat matrix
    std::vector<double> data(n * m, 0.0f);
    for (int i = 0; i < n; ++i) {
        data[i * m + 0] = 0.98 * (i + 1) * (i + 1) / (double)(n * n);  // boundary
        for (int j = 1; j < m; ++j)
            data[i * m + j] = ((double)(m - j) * (m - j)) / (double)(m * m);  // interior
    }

    // Create CPU and GPU copies of the data
    std::vector<double> cpu_data = data;
    std::vector<double> gpu_data = data;
    std::vector<double> cpu_avg(n, 0.0), gpu_avg(n, 0.0);

    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;

    // Use stopping only for GPU
    bool use_stop_cpu = false;
    bool use_stop_gpu = use_stop_flag;

    // Run CPU simulation (unless disabled)
    if (use_cpu) {
        std::cout << "🔁 Running CPU simulation...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_heat_propagation(cpu_data.data(), n, m, p, use_stop_cpu, stop_avg, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[CPU] Total Time: " << cpu_time_ms << " ms\n";
    }

    // Run GPU simulation
    std::cout << "🔁 Running GPU simulation...\n";
    auto t0_gpu = std::chrono::high_resolution_clock::now();
    launch_cuda_heat(gpu_data.data(), n, m, p, use_stop_gpu, stop_avg, show_timing);
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    gpu_time_ms = std::chrono::duration<double, std::milli>(t1_gpu - t0_gpu).count();

    // Display first 3 values of each row (first 5 rows only)
    std::cout << "First 3 values of each row:\n";
    for (int i = 0; i < std::min(n, 5); ++i) {
        for (int j = 0; j < std::min(m, 3); ++j)
            std::cout << gpu_data[i * m + j] << " ";
        std::cout << "\n";
    }

    // Compare CPU and GPU results (if CPU is enabled)
    if (use_cpu) {
        std::cout << "🔍 Comparing CPU and GPU results after " << p << " steps...\n";
        compare_results(cpu_data.data(), gpu_data.data(), n * m,
                        cpu_avg.data(), gpu_avg.data(), n);
        print_compare_debug(cpu_data.data(), gpu_data.data(), n, m);

        // Show speedup1
        std::cout << "🚀 Speedup (CPU Time / GPU Time): "
                  << cpu_time_ms / gpu_time_ms << "x\n";
    }

    return 0;
}
