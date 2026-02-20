#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 50000

__global__ void mandelbrot_kernel(long long *results, int width, int height, int max_iter) {
    // RESTORE THESE LINES: These define which pixel this thread handles
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // All math converted to float (FP32) for Orin performance
        float cr = (float)(col - width / 1.5f) / (width / 3.0f);
        float ci = (float)(row - height / 2.0f) / (height / 3.0f);

        float zr = 0.0f, zi = 0.0f;
        long long iter_count = 0;

        for (int i = 0; i < max_iter; i++) {
            float zr_sq = zr * zr;
            float zi_sq = zi * zi;
            
            // Optimization: check escape condition before calculating new values
            if (zr_sq + zi_sq > 4.0f) break;

            zi = 2.0f * zr * zi + ci;
            zr = zr_sq - zi_sq + cr;
            iter_count++;
        }

        results[row * width + col] = iter_count;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(long long);
    std::vector<long long> h_results(WIDTH * HEIGHT);

    long long *d_results;
    cudaMalloc(&d_results, size);

    // 16x16 threads per block is a sweet spot for Orin/Ampere
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Running on Jetson Orin GPU (FP32 Mode)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_results, WIDTH, HEIGHT, MAX_ITER);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_results.data(), d_results, size, cudaMemcpyDeviceToHost);

    long long final_checksum = 0;
    for (long long val : h_results) final_checksum += val;

    std::chrono::duration<double> diff = end - start;
    std::cout << "\n--- CUDA GPU Results ---" << std::endl;
    std::cout << "Total Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Verification Checksum: " << final_checksum << std::endl;

    cudaFree(d_results);
    return 0;
}