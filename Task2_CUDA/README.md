# Task 2 – Parallel Heat Propagation (CUDA)

## 📌 Description

This task implements a 2D heat propagation model using CUDA. The simulation computes the temperature evolution across a grid over multiple time steps. Shared memory optimization is applied in the GPU kernel that calculates per-row averages.

## ✨ Key Features

- CUDA parallelization of the heat update kernel.
- Shared memory optimization for row-wise averaging.
- Early stopping when average row temperature exceeds a threshold.
- Performance timing using CUDA Events.
- CPU–GPU result comparison with detailed mismatch debug output.

## 🛠️ Build

Run `make` to compile:

This compiles:

- `main.cpp`
- `heat_utils.cpp`, `heat_utils.h`
- `heat_kernel.cu`

## 🚀 Usage

```bash
./task2 [options]
```

| Option       | Description                                            |
| ------------ | ------------------------------------------------------ |
| `-n <rows>`  | Number of rows in the matrix (default: 64)             |
| `-m <cols>`  | Number of columns in the matrix (default: 64)          |
| `-p <steps>` | Number of time steps (default: 100)                    |
| `-a <value>` | Enable early stop if row average ≥ value               |
| `-t`         | Print timing information (H2D/D2H, kernel, avg, total) |
| `-c`         | Skip CPU simulation (run GPU only)                     |

## ✅ Example Tests

### 1. Correctness Verification

Run both CPU and GPU versions for 10 steps and compare results:

```bash
./task2 -n 512 -m 512 -p 10 -t
```

### 2. Performance Benchmark

Run a large grid with early stopping (skip CPU):

```bash
./task2 -n 2048 -m 2048 -p 500 -a 0.01 -t -c
```

### 3. ❌ Mismatch Warning Example

This causes mismatch (GPU exits early, CPU runs full steps):

```bash
./task2 -n 2048 -m 2048 -p 5 -a 0.01 -t
```

## 📤 Output Format

When `-t` is used, output includes:

```txt
[GPU] Memcpy H2D: 2.930752 ms
[GPU] Propagation Time: 28.511366 ms
[GPU] Row Average Time: 0.161534 ms
[GPU] Memcpy D2H: 3.022390 ms
[GPU] Total Time: 34.740213 ms
```

If early stop is triggered (`-a`), output includes:

```txt
🔴 Stopped at iteration 1 (avg ≥ 0.010)
```

## 🧠 Shared Memory Optimization

- `row_avg_kernel` uses shared memory for per-row reduction.
- Debug print confirms kernel is launched and executed:

row_avg_kernel launch succeeded.