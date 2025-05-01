# Task 1 – CPU Implementation of Heat Propagation

This task implements a 2D heat propagation simulation using finite differences on the CPU. The propagation occurs **horizontally only**, as per assignment specification. The program supports custom matrix dimensions, iteration count, and early-stopping via average temperature threshold.

## ✅ Features

- Supports arbitrary matrix sizes (`n x m`)
- Horizontal heat diffusion using given weighted formula
- Edge handling with fixed left-boundary initialization
- Optional early termination based on row average temperature (`-a`)
- Simple command-line interface

## ⚙️ Compilation

Use the provided Makefile:

```bash
make
```



This generates an executable `task1_cpu`.

## 🚀 Usage

```bash
./task1_cpu [-n <rows>] [-m <cols>] [-p <iterations>] [-a <avg-threshold>]
```

### Arguments

| Flag | Description                                     | Default |
| ---- | ----------------------------------------------- | ------- |
| `-n` | Number of rows (pipes in the radiator)          | 32      |
| `-m` | Number of columns (temperature points per pipe) | 32      |
| `-p` | Number of propagation iterations                | 10      |
| `-a` | (Optional) Row average temperature threshold    | None    |



If the `-a` option is used, the program will terminate early if **any row** exceeds this average temperature.

## 🧪 Example Runs

```bash
# Run with default settings
./task1_cpu

# Simulate 64x128 grid with 50 iterations
./task1_cpu -n 64 -m 128 -p 50

# Run with early stop threshold
./task1_cpu -n 64 -m 128 -p 100 -a 0.7
```

### Sample Output

```
🛑 Stopped early at iteration 23 due to average temperature >= 0.7
Row 0: 0.00231 0.94213 0.53311
Row 1: 0.00583 0.94831 0.54421
...
```

## 📌 Notes

- The propagation is horizontal only; vertical elements are unaffected.

- Memory is dynamically allocated and reused via double buffering.

- Left-most column values are fixed and initialized to:

  ```cpp
  matrix[i][0] = 0.98 * (i+1)^2 / (n^2)
  ```

## 📦 Files

- `task1_cpu.cpp`: Main simulation logic
- `Makefile`: Build script
- `README.md`: This documentation