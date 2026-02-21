#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 10000
#define TILE_SIZE 16
#define STRIPS 4 // Divide image into 4 horizontal strips to reset watchdog

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__ void mandelbrot_kernel(int* output, int width, int height, int start_row, int strip_height, float x_min, float y_min, float dx, float dy) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int local_row = blockIdx.y * blockDim.y + ty;
    int global_row = start_row + local_row;

    if (col < width && local_row < strip_height && global_row < height) {
        float x0 = x_min + (float)col * dx;
        float y0 = y_min + (float)global_row * dy;
        float x = 0.0f, y = 0.0f, x2 = 0.0f, y2 = 0.0f;
        int iter = 0;

        while (x2 + y2 <= 4.0f && iter < MAX_ITER) {
            y = 2.0f * x * y + y0;
            x = x2 - y2 + x0;
            x2 = x * x;
            y2 = y * y;
            iter++;
        }
        output[global_row * width + col] = iter;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on Device: " << prop.name << std::endl;

    float x_min = -2.0f, x_max = 1.0f, y_min = -1.5f, y_max = 1.5f;
    float dx = (x_max - x_min) / WIDTH;
    float dy = (y_max - y_min) / HEIGHT;

    size_t total_size = WIDTH * HEIGHT * sizeof(int);
    std::vector<int> h_results(WIDTH * HEIGHT);
    int *d_results;
    checkCudaError(cudaMalloc(&d_results, total_size), "cudaMalloc");

    int strip_height = HEIGHT / STRIPS;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks( (WIDTH + TILE_SIZE - 1) / TILE_SIZE, (strip_height + TILE_SIZE - 1) / TILE_SIZE );

    std::cout << "Calculating Mandelbrot in " << STRIPS << " strips..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < STRIPS; i++) {
        int start_row = i * strip_height;
        mandelbrot_kernel<<<blocks, threads>>>(d_results, WIDTH, HEIGHT, start_row, strip_height, x_min, y_min, dx, dy);
        
        // This synchronization is the secret sauce for the Nano.
        // It tells the CPU to wait, which resets the OS watchdog timer.
        checkCudaError(cudaDeviceSynchronize(), "Stripe Sync");
    }

    auto end = std::chrono::high_resolution_clock::now();
    checkCudaError(cudaMemcpy(h_results.data(), d_results, total_size, cudaMemcpyDeviceToHost), "cudaMemcpy");

    long long checksum = 0;
    for (int val : h_results) checksum += val;

    std::cout << "Total Time: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
    std::cout << "Verification Checksum: " << checksum << std::endl;

    cudaFree(d_results);
    return 0;
}