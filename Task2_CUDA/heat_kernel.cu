#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// CUDA kernel for 1D heat propagation (row-wise only)
// CUDA kernel for 2D heat propagation with 5-point stencil (row-wise only)
__global__ void heat_kernel(float* next, const float* prev, int n, int m) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= n || j >= m) return;

    int idx = i * m + j;

    // First: copy all values by default (makes it equivalent to CPU)
    next[idx] = prev[idx];

    // Then: only stencil region will overwrite with new value
    if (i >= 1 && i < n - 1 && j >= 2 && j < m - 2) {
        next[idx] =
            (1.60f * prev[i * m + (j - 2)] +
             1.55f * prev[i * m + (j - 1)] +
             1.00f * prev[i * m + j]     +
             0.60f * prev[i * m + (j + 1)] +
             0.25f * prev[i * m + (j + 2)]) / 5.0f;
    }
}




// CUDA kernel to compute the average temperature of each row
__global__ void row_avg_kernel(const float* data, float* row_avg, int n, int m) {
    int i = blockIdx.x;         // Each block handles one row
    int j = threadIdx.x;        // Each thread processes one column in the row

    if (i >= n || j >= m) return;

    // Shared memory to accumulate sum per row
    __shared__ float local_sum[1];

    // Initialize shared memory (only thread 0)
    if (j == 0) local_sum[0] = 0.0f;
    __syncthreads();

    // Each thread adds its column's value to shared row sum
    atomicAdd(&local_sum[0], data[i * m + j]);
    __syncthreads();

    // First thread in the block writes the average
    if (j == 0) {
        row_avg[i] = local_sum[0] / m;
    }
}



// Host function to perform heat propagation using GPU
extern "C" void launch_cuda_heat(float* host_prev, int n, int m, int p, bool use_stop, float stop_avg, bool show_timing) {
    // Allocate device pointers
    float *d_prev, *d_next, *d_avg;

    // CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_avg_kernel, stop_avg_kernel;
    cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h;

    // Create all timing events
    cudaEventCreate(&start_total);     cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_avg_kernel);cudaEventCreate(&stop_avg_kernel);
    cudaEventCreate(&start_h2d);       cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_d2h);       cudaEventCreate(&stop_d2h);

    // Start total execution timer
    cudaEventRecord(start_total);

    // Allocate memory on GPU
    cudaMalloc(&d_prev, n * m * sizeof(float));
    cudaMalloc(&d_next, n * m * sizeof(float));
    cudaMalloc(&d_avg, n * sizeof(float));

    // Copy initial data to GPU and time it
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_prev, host_prev, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    // Kernel launch configuration
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    // Allocate host-side buffer for row averages
    std::vector<float> h_avg(n);

    float total_avg_time = 0.0f;

    // Start kernel timing
    cudaEventRecord(start_kernel);

    for (int step = 0; step < p; ++step) {
        heat_kernel<<<grid, block>>>(d_next, d_prev, n, m);
        //  Add this immediately after kernel launch
        cudaDeviceSynchronize();
        std::swap(d_prev, d_next);

        if (use_stop) {
            // Time row average calculation
            cudaEventRecord(start_avg_kernel);
            row_avg_kernel<<<n, m>>>(d_prev, d_avg, n, m);
            cudaEventRecord(stop_avg_kernel);
            cudaEventSynchronize(stop_avg_kernel);

            float avg_time = 0.0f;
            cudaEventElapsedTime(&avg_time, start_avg_kernel, stop_avg_kernel);
            total_avg_time += avg_time;

            // Copy row averages back and check stopping condition
            cudaMemcpy(h_avg.data(), d_avg, n * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; ++i) {
                if (h_avg[i] >= stop_avg) {
                    std::cout << "🔴 Stopped at iteration " << step + 1
                              << " due to average temp >= " << stop_avg << "\n";
                    goto END_KERNEL;
                }
            }
        }
    }

END_KERNEL:
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // Copy final matrix back to host
    cudaEventRecord(start_d2h);
    cudaMemcpy(host_prev, d_prev, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);

    // Stop total timer
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Compute elapsed times
    float t_total, t_kernel, t_avg, t_h2d, t_d2h;
    cudaEventElapsedTime(&t_total, start_total, stop_total);
    cudaEventElapsedTime(&t_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&t_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&t_d2h, start_d2h, stop_d2h);
    t_avg = total_avg_time;

    if (show_timing) {
        std::cout << "[GPU] Memcpy H2D:       " << t_h2d    << " ms\n";
        std::cout << "[GPU] Propagation Time: " << t_kernel << " ms\n";
        std::cout << "[GPU] Row Average Time: " << t_avg    << " ms\n";
        std::cout << "[GPU] Memcpy D2H:       " << t_d2h    << " ms\n";
        std::cout << "[GPU] Total Time:       " << t_total  << " ms\n";
    }

    // Cleanup
    cudaFree(d_prev); cudaFree(d_next); cudaFree(d_avg);
    cudaEventDestroy(start_total);     cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_avg_kernel);cudaEventDestroy(stop_avg_kernel);
    cudaEventDestroy(start_h2d);       cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_d2h);       cudaEventDestroy(stop_d2h);
}


