#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 50000

int main() {
    queue q(gpu_selector_v); 
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << " (FP32)\n";

    std::vector<long long> results(WIDTH * HEIGHT, 0);
    auto start_time = std::chrono::high_resolution_clock::now();

    {
        buffer<long long, 1> buf_res(results.data(), range<1>(WIDTH * HEIGHT));

        q.submit([&](handler& h) {
            accessor acc(buf_res, h, write_only);

            h.parallel_for(range<2>(HEIGHT, WIDTH), [=](id<2> idx) {
                int row = idx[0];
                int col = idx[1];

                // Switched math to float
                float cr = (float)(col - WIDTH / 1.5f) / (WIDTH / 3.0f);
                float ci = (float)(row - HEIGHT / 2.0f) / (HEIGHT / 3.0f);

                float zr = 0.0f, zi = 0.0f;
                long long iter_count = 0;

                for (int i = 0; i < MAX_ITER; i++) {
                    float zr_sq = zr * zr;
                    float zi_sq = zi * zi;
                    
                    if (zr_sq + zi_sq > 4.0f) break;

                    zi = 2.0f * zr * zi + ci;
                    zr = zr_sq - zi_sq + cr;
                    iter_count++;
                }
                acc[row * WIDTH + col] = iter_count;
            });
        });
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    long long final_checksum = 0;
    for (auto& val : results) final_checksum += val;

    std::cout << "\n--- SYCL GPU FP32 Results ---\n";
    std::cout << "Total Time: " << diff.count() << " seconds\n";
    std::cout << "Verification Checksum: " << final_checksum << "\n";

    return 0;
}