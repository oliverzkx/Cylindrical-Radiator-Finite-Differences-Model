#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>  // for getopt (command line argument parsing)
#include <chrono>

#define DEFAULT_N 32
#define DEFAULT_M 32
#define DEFAULT_P 10

int main(int argc, char* argv[]) {
    // Matrix dimensions and propagation steps
    int n = DEFAULT_N, m = DEFAULT_M, p = DEFAULT_P;

    // Optional stopping condition: average temperature threshold
    bool use_stop = false;
    float stop_avg = 0.0f;

    // Parse command-line arguments: -n, -m, -p, -a
    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:a:")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': use_stop = true; stop_avg = atof(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-n rows] [-m cols] [-p iterations] [-a avg_stop]\n";
                return 1;
        }
    }

    // Allocate two 2D float matrices using std::vector
    std::vector<std::vector<float>> prev(n, std::vector<float>(m));
    std::vector<std::vector<float>> next(n, std::vector<float>(m));

    // Initialize the matrix:
    // - Column 0 with boundary condition
    // - Other columns with initial temperature values
    for (int i = 0; i < n; ++i) {
        prev[i][0] = 0.98f * (i + 1) * (i + 1) / (float)(n * n);  // Boundary value in column 0
        for (int j = 1; j < m; ++j) {
            prev[i][j] = (float)((m - j) * (m - j)) / (m * m);     // Initial values for other columns
        }
    }

    // Inform the user
    std::cout << "Matrix initialized. Size: " << n
              << " x " << m << ", iterations: " << p << "\n";

    //finite difference update
    for (int step = 0; step < p; ++step) {
        for (int u = 0; u < n; ++u) {
            for (int j = 2; j < m - 2; ++j) {
                next[u][j] = (
                    1.60f * prev[u][j - 2] +
                    1.55f * prev[u][j - 1] +
                    1.00f * prev[u][j] +
                    0.60f * prev[u][j + 1] +
                    0.25f * prev[u][j + 2]
                ) / 5.0f;
            }

            // Copy fixed values at boundaries
            next[u][0] = prev[u][0];
            next[u][1] = prev[u][1];
            next[u][m - 2] = prev[u][m - 2];
            next[u][m - 1] = prev[u][m - 1];
        }

        // Swap matrices for next iteration
        std::swap(prev, next);
        // If -a option was used, calculate average temp for each row
        if (use_stop) {
            bool stop = false;

            for (int u = 0; u < n; ++u) {
                float row_sum = 0.0f;
                for (int j = 0; j < m; ++j) {
                    row_sum += prev[u][j];
                }
                float row_avg = row_sum / m;

                if (row_avg >= stop_avg) {
                    stop = true;
                    break;
                }
            }

            if (stop) {
                std::cout << "🛑 Stopped early at iteration " << step + 1
                          << " due to average temperature >= " << stop_avg << "\n";
                break;
            }
        }
    }

    std::cout << "Final state (first 3 values of each row):\n";
    for (int i = 0; i < std::min(n, 5); ++i) {
        for (int j = 0; j < std::min(m, 3); ++j) {
            std::cout << prev[i][j] << " ";
        }
        std::cout << "\n";
    }



    return 0;
}
