# Heat Propagation Simulation 

This assignment explores 2D heat propagation using both CPU and CUDA GPU implementations. It consists of four tasks, each progressively introducing new concepts and optimizations. Each task has its own folder, codebase, and report.

------

## 🧩 Task Overview

### 🔹 `Task1_CPU/`

> Basic CPU version of heat propagation using iterative stencil computation.

- **task1_cpu.cc** – CPU-only implementation of the heat propagation logic.
- **Makefile** – Compiles the CPU version.
- **README.md** – Describes the logic, early stop strategy, and test results.

------

### 🔹 `Task2_CUDA/`

> First CUDA implementation: 2D kernel for heat propagation, row-wise average, and comparison with CPU.

- **heat_kernel.cu** – CUDA kernel to simulate heat propagation.
- **main.cpp** – Entry point; runs both CPU and GPU code and compares results.
- **heat_utils.\*`** – Shared helper code for timing and result comparison.
- **Makefile** – Builds CUDA executable.
- **README.md** – Includes kernel design, speedup tests, and observations.

------

### 🔹 `Task3_CUDA/`

> Optimized CUDA version using shared memory for faster per-row reduction and grid size benchmarking.

- **Shared Memory Optimization**: Significantly improves row-wise average performance.
- **main.cpp** – Measures GPU performance under varying thread block sizes (e.g. `32x32`, `64x4`).
- **output.png** – Benchmark chart comparing total time and speedup vs. grid size.
- **README.md** – Records experiments, grid search results, and analysis.

------

### 🔹 `Task4_CUDA/`

> Double precision version of Task 3 to evaluate accuracy and performance trade-offs.

- **All float types converted to double** to ensure higher accuracy.
- **task4_speedup.png** – Line chart comparing CPU-GPU speedup for different problem sizes.
- **README.md** – Compares precision, speedup, and block size performance for double precision.

------

## 📦 Compilation

Each folder includes a `Makefile` to build its executable:

```
make         # builds task1/task2/task3/task4 depending on folder
./taskX ...  # run with specified flags (-n, -m, -p, -t, etc.)
```

## 📊 Measurement Flags

- `-n`, `-m`: matrix dimensions (rows × cols)
- `-p`: number of iterations
- `-c`: disable CPU comparison
- `-t`: enable timing output (GPU & CPU)

------

## 📁 Notes

- Accurate numerical results and speedups are discussed in each task’s individual `README.md`.
- CUDA performance is tested with multiple thread block sizes and compared to CPU baselines.
- Task 3 and 4 include visual benchmarks (speedup plots).