#include <iostream>
#include <vector>
#include <getopt.h>

// Declare the CUDA interface function (defined in heat_kernel.cu)
extern void launch_cuda_heat(float* host_prev, int n, int m, int p, bool use_stop, float stop_avg);

int main(int argc, char* argv[]) {
    // Default simulation parameters
    int n = 32, m = 32, p = 10;
    bool cpu_mode = false, use_stop = false;
    float stop_avg = 0.0f;

    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:a:c")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': use_stop = true; stop_avg = atof(optarg); break;
            case 'c': cpu_mode = true; break;  // CPU mode flag (optional, not used here)
        }
    }

    // Initialize the temperature matrix with boundary in column 0
    std::vector<float> prev(n * m, 0.0f);
    for (int i = 0; i < n; ++i)
        prev[i * m + 0] = 0.98f * ((i + 1) * (i + 1)) / (float)(n * n);

    // GPU heat propagation
    if (!cpu_mode)
        launch_cuda_heat(prev.data(), n, m, p, use_stop, stop_avg);
    else
        std::cout << "[CPU mode is not implemented in this version]\n";

    // Output first 3 values of each row for inspection
    for (int i = 0; i < n && i < 5; ++i)
        std::cout << prev[i * m + 0] << " " << prev[i * m + 1] << " " << prev[i * m + 2] << "\n";

    return 0;
}
