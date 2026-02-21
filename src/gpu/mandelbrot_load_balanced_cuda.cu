#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 50000

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void mandelbrot_kernel(long long *results, int width, int height, int max_iter) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float cr = (float)(col - width / 1.5f) / (width / 3.0f);
        float ci = (float)(row - height / 2.0f) / (height / 3.0f);

        float zr = 0.0f, zi = 0.0f;
        long long iter_count = 0;

        for (int i = 0; i < max_iter; i++) {
            float zr_sq = zr * zr;
            float zi_sq = zi * zi;
            
            if (zr_sq + zi_sq > 4.0f) break;

            zi = 2.0f * zr * zi + ci;
            zr = zr_sq - zi_sq + cr;
            iter_count++;
        }
        results[row * width + col] = iter_count;
    }
}

int main() {
    // Determine Device Name for better logging
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on Device: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;

    size_t size = WIDTH * HEIGHT * sizeof(long long);
    std::vector<long long> h_results(WIDTH * HEIGHT, 0);

    long long *d_results;
    
    // 1. Allocate Memory with error check
    checkCudaError(cudaMalloc(&d_results, size), "cudaMalloc");

    // 2. Define Grid/Block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Calculating Mandelbrot (" << WIDTH << "x" << HEIGHT << ")..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();

    // 3. Launch Kernel
    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_results, WIDTH, HEIGHT, MAX_ITER);
    
    // 4. Check for immediate launch errors (like architecture mismatch)
    checkCudaError(cudaGetLastError(), "Kernel Launch");

    // 5. Synchronize and check for runtime errors
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();

    // 6. Copy back to Host
    checkCudaError(cudaMemcpy(h_results.data(), d_results, size, cudaMemcpyDeviceToHost), "cudaMemcpy");

    // Calculate Checksum
    long long final_checksum = 0;
    for (long long val : h_results) final_checksum += val;

    std::chrono::duration<double> diff = end - start;
    std::cout << "\n--- CUDA GPU Results ---" << std::endl;
    std::cout << "Total Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Verification Checksum: " << final_checksum << std::endl;

    // 7. Cleanup
    cudaFree(d_results);
    
    return 0;
}