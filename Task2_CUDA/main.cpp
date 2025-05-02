#include <iostream>
#include <vector>
#include <getopt.h>
#include <chrono>
using namespace std::chrono;

#define DEFAULT_N 32
#define DEFAULT_M 32
#define DEFAULT_P 10


// Declare the CUDA interface function (defined in heat_kernel.cu)
extern void launch_cuda_heat(float* host_prev, int n, int m, int p,
                             bool use_stop, float stop_avg, bool show_timing);

void cpu_heat_propagation(float* data, int n, int m, int p, bool use_stop, float stop_avg, bool show_timing) {
    std::vector<float> prev(data, data + n * m);
    std::vector<float> next(n * m, 0.0f);

    auto start = high_resolution_clock::now();

    for (int step = 0; step < p; ++step) {
        for (int u = 0; u < n; ++u) {
            for (int j = 2; j < m - 2; ++j) {
                next[u * m + j] = (
                    1.60f * prev[u * m + j - 2] +
                    1.55f * prev[u * m + j - 1] +
                    0.60f * prev[u * m + j + 1] +
                    0.25f * prev[u * m + j + 2]
                ) / 5.0f;
            }

            // Preserve fixed boundary values
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
                    std::cout << "🛑 CPU stopped early at iteration " << step + 1
                              << " due to average temp >= " << stop_avg << "\n";
                    break;
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "[CPU] Propagation Time: " << duration << " ms\n";

    // Copy final state back
    std::copy(prev.begin(), prev.end(), data);
}


int main(int argc, char* argv[]) {
    int n = DEFAULT_N, m = DEFAULT_M, p = DEFAULT_P;
    bool use_stop = false, cpu_mode = false, show_timing = false;
    float stop_avg = 0.0f;

    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:a:ct")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': use_stop = true; stop_avg = atof(optarg); break;
            case 'c': cpu_mode = true; break;
            case 't': show_timing = true; break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-n rows] [-m cols] [-p steps] [-a avg_stop] [-c] [-t]\n";
                return 1;
        }
    }

    // Allocate data
    std::vector<float> data(n * m);

    // Initialize data
    for (int i = 0; i < n; ++i) {
        data[i * m + 0] = 0.98f * (i + 1) * (i + 1) / (float)(n * n);
        for (int j = 1; j < m; ++j) {
            data[i * m + j] = (float)((m - j) * (m - j)) / (float)(m * m);
        }
    }

    if (cpu_mode) {
        cpu_heat_propagation(data.data(), n, m, p, use_stop, stop_avg, show_timing);
    } else {
        launch_cuda_heat(data.data(), n, m, p, use_stop, stop_avg, show_timing);
    }

    // Print final result (first 3 columns of first few rows)
    std::cout << "First 3 values of each row:\n";
    for (int i = 0; i < std::min(n, 5); ++i) {
        for (int j = 0; j < std::min(m, 3); ++j) {
            std::cout << data[i * m + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

