#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// CUDA kernel for 1D heat propagation (row-wise only)
__global__ void heat_kernel(float* prev, float* next, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col < 2 || col >= m - 2) return;

    int idx = row * m + col;
    next[idx] = (1.60f * prev[idx - 2] +
                 1.55f * prev[idx - 1] +
                 0.60f * prev[idx + 1] +
                 0.25f * prev[idx + 2]) / 5.0f;
}

// CUDA kernel to compute the average temperature of each row
__global__ void row_avg_kernel(float* data, float* row_avg, int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < m; ++j)
        sum += data[row * m + j];

    row_avg[row] = sum / m;
}

// Host function to perform heat propagation using GPU
extern "C" void launch_cuda_heat(float* host_prev, int n, int m, int p, bool use_stop, float stop_avg, bool show_timing) {
    float *d_prev, *d_next, *d_avg;
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_avg_kernel, stop_avg_kernel;
    cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h;

    // Create events for timing
    cudaEventCreate(&start_total);  cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel); cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_avg_kernel); cudaEventCreate(&stop_avg_kernel);
    cudaEventCreate(&start_h2d);    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_d2h);    cudaEventCreate(&stop_d2h);

    // Start total timer
    cudaEventRecord(start_total);

    // Allocate memory
    cudaMalloc(&d_prev, n * m * sizeof(float));
    cudaMalloc(&d_next, n * m * sizeof(float));
    cudaMalloc(&d_avg, n * sizeof(float));

    // H2D timing
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_prev, host_prev, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    // Grid setup
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (n + 15) / 16);
    std::vector<float> h_avg(n);

    float total_avg_time = 0.0f;

    // Start kernel timing
    cudaEventRecord(start_kernel);

    for (int step = 0; step < p; ++step) {
        heat_kernel<<<grid, block>>>(d_prev, d_next, n, m);
        std::swap(d_prev, d_next);

        if (use_stop) {
            // Independent timing for row average kernel
            cudaEventRecord(start_avg_kernel);
            row_avg_kernel<<<(n + 255) / 256, 256>>>(d_prev, d_avg, n, m);
            cudaEventRecord(stop_avg_kernel);
            cudaEventSynchronize(stop_avg_kernel);

            float avg_time = 0.0f;
            cudaEventElapsedTime(&avg_time, start_avg_kernel, stop_avg_kernel);
            total_avg_time += avg_time;

            cudaMemcpy(h_avg.data(), d_avg, n * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; ++i) {
                if (h_avg[i] >= stop_avg) {
                    std::cout << "🛑 Stopped at iteration " << step + 1
                              << " due to average temp >= " << stop_avg << "\n";
                    goto END_KERNEL;
                }
            }
        }
    }

END_KERNEL:
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // D2H timing
    cudaEventRecord(start_d2h);
    cudaMemcpy(host_prev, d_prev, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);

    // Stop total timer
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Read elapsed times
    float t_total, t_kernel, t_avg, t_h2d, t_d2h;
    cudaEventElapsedTime(&t_total, start_total, stop_total);
    cudaEventElapsedTime(&t_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&t_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&t_d2h, start_d2h, stop_d2h);
    t_avg = total_avg_time;

    // Output timing results
    std::cout << "[GPU] Memcpy H2D:       " << t_h2d    << " ms\n";
    std::cout << "[GPU] Propagation Time: " << t_kernel << " ms\n";
    std::cout << "[GPU] Row Average Time: " << t_avg    << " ms\n";
    std::cout << "[GPU] Memcpy D2H:       " << t_d2h    << " ms\n";
    std::cout << "[GPU] Total Time:       " << t_total  << " ms\n";

    // Cleanup
    cudaFree(d_prev); cudaFree(d_next); cudaFree(d_avg);
    cudaEventDestroy(start_total);  cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_avg_kernel); cudaEventDestroy(stop_avg_kernel);
    cudaEventDestroy(start_h2d);    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_d2h);    cudaEventDestroy(stop_d2h);
}

