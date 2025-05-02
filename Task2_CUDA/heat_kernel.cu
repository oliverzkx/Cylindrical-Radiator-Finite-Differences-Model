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


/* ========================================================================== */
/* 1. row_avg_kernel : shared‑memory reduction that covers ALL m columns      */
/* ========================================================================== */
__global__ void row_avg_kernel(const float* __restrict__ data,
                               float*       __restrict__ row_avg,
                               int n, int m)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;           // one block == one row
    int tid = threadIdx.x;          // 0 .. 1023

    // ----- 1) stride‑load this row into a private register ------------------
    float sum = 0.0f;
    for (int col = tid; col < m; col += blockDim.x)
        sum += data[row * m + col];

    // ----- 2) reduction in shared memory -----------------------------------
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) row_avg[row] = sdata[0] / m;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(">>> row_avg_kernel is running on device.\n");
    }
}

/* ========================================================================== */
/* 2. launch_cuda_heat                                                        */
/* ========================================================================== */
extern "C"
void launch_cuda_heat(float* host_prev,
                      int   n,  int   m,  int p,
                      bool  use_stop, float stop_avg,
                      bool  show_timing)
{
    /* -------- device buffers -------- */
    float *d_prev, *d_next, *d_avg;
    cudaMalloc(&d_prev, n * m * sizeof(float));
    cudaMalloc(&d_next, n * m * sizeof(float));
    cudaMalloc(&d_avg , n       * sizeof(float));

    /* -------- CUDA events -------- */
    cudaEvent_t eTotS,eTotE,eKerS,eKerE,eAvgS,eAvgE,eH2DS,eH2DE,eD2HS,eD2HE;
    #define NEW(ev) cudaEventCreate(&(ev))
    NEW(eTotS); NEW(eTotE); NEW(eKerS); NEW(eKerE);
    NEW(eAvgS); NEW(eAvgE); NEW(eH2DS); NEW(eH2DE); NEW(eD2HS); NEW(eD2HE);
    #undef NEW

    cudaEventRecord(eTotS);

    /* -------- H2D copy -------- */
    cudaEventRecord(eH2DS);
    cudaMemcpy(d_prev, host_prev, n*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(eH2DE);  cudaEventSynchronize(eH2DE);

    /* -------- kernel configs -------- */
    dim3 blkProp(16,16);
    dim3 grdProp((m + blkProp.x - 1)/blkProp.x,
                 (n + blkProp.y - 1)/blkProp.y);

    const int threadsPerRow = 1024;                 // fixed upper–limit
    const size_t shBytes    = threadsPerRow * sizeof(float);

    std::vector<float> h_avg(n);
    float totalAvgMs = 0.f;

    cudaEventRecord(eKerS);

    for (int step = 0; step < p; ++step) {
        heat_kernel<<<grdProp, blkProp>>>(d_next, d_prev, n, m);
        cudaDeviceSynchronize();
        std::swap(d_prev, d_next);

        if (!use_stop) continue;

        /* ---- row‑average timing + kernel ---- */
        cudaEventRecord(eAvgS);

        row_avg_kernel<<<n, threadsPerRow, shBytes>>>(d_prev, d_avg, n, m);

        // 🔍 检查 kernel 启动是否成功
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("❌ row_avg_kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        } else {
            // 可选调试输出
             printf("✅ row_avg_kernel launch succeeded.\n");
        }

        cudaEventRecord(eAvgE);
        cudaEventSynchronize(eAvgE);  // ensure stop‑event written

        float t;
        cudaEventElapsedTime(&t, eAvgS, eAvgE);
        totalAvgMs += t;

        cudaMemcpy(h_avg.data(), d_avg, n*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            if (h_avg[i] >= stop_avg) {
                printf("🛑 Stopped at iteration %d (avg ≥ %.3f)\n", step+1, stop_avg);
                step = p;  // exit loop
                break;
            }
        }
    }

    cudaEventRecord(eKerE);  cudaEventSynchronize(eKerE);

    /* -------- D2H copy -------- */
    cudaEventRecord(eD2HS);
    cudaMemcpy(host_prev, d_prev, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(eD2HE);  cudaEventSynchronize(eD2HE);

    cudaEventRecord(eTotE);  cudaEventSynchronize(eTotE);

    /* -------- timing report -------- */
    float tTot,tKer,tH2D,tD2H;
    cudaEventElapsedTime(&tTot,eTotS,eTotE);
    cudaEventElapsedTime(&tKer,eKerS,eKerE);
    cudaEventElapsedTime(&tH2D,eH2DS,eH2DE);
    cudaEventElapsedTime(&tD2H,eD2HS,eD2HE);

    if (show_timing) {
        printf("[GPU] Memcpy H2D:       %.6f ms\n", tH2D);
        printf("[GPU] Propagation Time: %.6f ms\n", tKer);
        printf("[GPU] Row Average Time: %.6f ms\n", totalAvgMs);
        printf("[GPU] Memcpy D2H:       %.6f ms\n", tD2H);
        printf("[GPU] Total Time:       %.6f ms\n", tTot);
    }

    /* -------- cleanup -------- */
    cudaFree(d_prev); cudaFree(d_next); cudaFree(d_avg);
    #define DEL(ev) cudaEventDestroy(ev)
    DEL(eTotS); DEL(eTotE); DEL(eKerS); DEL(eKerE);
    DEL(eAvgS); DEL(eAvgE); DEL(eH2DS); DEL(eH2DE); DEL(eD2HS); DEL(eD2HE);
    #undef DEL
}

