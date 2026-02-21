#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sycl;

// --- Configuration ---
#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 500000
#define CHUNK_SIZE 10  // Process 10 rows at a time to stay under hangcheck limits

void print_progress(int current, int total) {
    float progress = (float)current / total;
    int barWidth = 50;
    std::cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << " rows)" << std::flush;
}

int main() {
    // 1. Setup Device and Queue
    queue q(gpu_selector_v);
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Iterations: " << MAX_ITER << " | Resolution: " << WIDTH << "x" << HEIGHT << "\n\n";

    // 2. Allocate Unified Shared Memory (USM) for the result
    int *output = malloc_shared<int>(WIDTH * HEIGHT, q);

    auto start_time = std::chrono::steady_clock::now();

    // 3. Row-by-Row Processing (Chunked)
    for (int start_row = 0; start_row < HEIGHT; start_row += CHUNK_SIZE) {
        int rows_to_process = std::min(CHUNK_SIZE, HEIGHT - start_row);

        // Submit a small "chunk" of rows to the GPU
        q.parallel_for(range<2>(rows_to_process, WIDTH), [=](id<2> index) {
            int local_row = index[0];
            int global_row = start_row + local_row;
            int x_idx = index[1];

            double x0 = -2.0 + 3.0 * x_idx / WIDTH;
            double y0 = -1.5 + 3.0 * global_row / HEIGHT;

            double x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0;
            int iter = 0;

            while (x2 + y2 <= 4.0 && iter < MAX_ITER) {
                y = 2.0 * x * y + y0;
                x = x2 - y2 + x0;
                x2 = x * x;
                y2 = y * y;
                iter++;
            }
            output[global_row * WIDTH + x_idx] = iter;
        }).wait(); // Crucial: wait for the chunk to finish so the driver can breathe

        print_progress(start_row + rows_to_process, HEIGHT);
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 4. Checksum Verification
    unsigned long long checksum = 0;
    for (int i = 0; i < WIDTH * HEIGHT; i++) checksum += output[i];

    std::cout << "\n\n--- SYCL GPU Results ---";
    std::cout << "\nTotal Time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds";
    std::cout << "\nChecksum:   " << checksum << "\n";

    // 5. Cleanup
    free(output, q);

    return 0;
}