#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 50000
#define TILE_SIZE 16

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__ void mandelbrot_tiled_fp32(int* output, int width, int height, float x_min, float y_min, float dx, float dy) {
    // Shared memory to cache tile constants in FP32
    __shared__ float tile_coords[2]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    if (tx == 0 && ty == 0) {
        tile_coords[0] = x_min + (float)(blockIdx.x * blockDim.x) * dx;
        tile_coords[1] = y_min + (float)(blockIdx.y * blockDim.y) * dy;
    }
    __syncthreads();

    if (col < width && row < height) {
        float x0 = tile_coords[0] + (float)tx * dx;
        float y0 = tile_coords[1] + (float)ty * dy;
        float x = 0.0f, y = 0.0f;
        int iter = 0;

        float x2 = 0.0f, y2 = 0.0f;

        // FP32 Math - Much faster on Jetson Nano/Orin
        while (x2 + y2 <= 4.0f && iter < MAX_ITER) {
            y = 2.0f * x * y + y0;
            x = x2 - y2 + x0;
            x2 = x * x;
            y2 = y * y;
            iter++;
        }
        output[row * width + col] = iter;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on Device: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;

    // Single Precision constants
    float x_min = -2.0f, x_max = 1.0f;
    float y_min = -1.5f, y_max = 1.5f;
    float dx = (x_max - x_min) / (float)WIDTH;
    float dy = (y_max - y_min) / (float)HEIGHT;

    size_t size = WIDTH * HEIGHT * sizeof(int);
    std::vector<int> h_results(WIDTH * HEIGHT, 0);

    int *d_results;
    checkCudaError(cudaMalloc(&d_results, size), "cudaMalloc");

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Calculating Mandelbrot FP32 (" << WIDTH << "x" << HEIGHT << ")..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();

    mandelbrot_tiled_fp32<<<numBlocks, threadsPerBlock>>>(d_results, WIDTH, HEIGHT, x_min, y_min, dx, dy);
    
    checkCudaError(cudaGetLastError(), "Kernel Launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();

    checkCudaError(cudaMemcpy(h_results.data(), d_results, size, cudaMemcpyDeviceToHost), "cudaMemcpy");

    long long final_checksum = 0;
    for (int val : h_results) final_checksum += (long long)val;

    std::chrono::duration<double> diff = end - start;
    std::cout << "\n--- CUDA GPU FP32 Results ---" << std::endl;
    std::cout << "Total Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Verification Checksum: " << final_checksum << std::endl;

    cudaFree(d_results);
    return 0;
}