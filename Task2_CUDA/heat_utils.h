#pragma once

void cpu_heat_propagation(float* data, int n, int m, int p,
                          bool use_stop, float stop_avg, bool show_timing);

void compare_results(const float* cpu_data, const float* gpu_data, int size,
                     const float* cpu_avg = nullptr, const float* gpu_avg = nullptr,
                     int rows = 0, float threshold = 1e-4f);

void compare_results_verbose(const float* cpu, const float* gpu, int size, float threshold = 1e-4f);