#pragma once

void cpu_heat_propagation(double* data, int n, int m, int p,
                          bool use_stop, double stop_avg, bool show_timing);

void compare_results(const double* cpu_data, const double* gpu_data, int size,
                     const double* cpu_avg = nullptr, const double* gpu_avg = nullptr,
                     int rows = 0, double threshold = 1e-4f);

void compare_results_verbose(const double* cpu, const double* gpu, int size, double threshold = 1e-4);

void print_compare_debug(const double* cpu, const double* gpu, int n, int m, double threshold = 1e-4);