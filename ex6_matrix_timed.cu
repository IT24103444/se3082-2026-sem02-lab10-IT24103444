#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define ROWS 10000
#define COLS 10000

__global__ void mulElem(const int *a, const int *b,
                        int *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
    size_t count = (size_t)ROWS * COLS;
    size_t size = count * sizeof(int);

    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    if (!a || !b || !c) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(0);

    for (size_t i = 0; i < count; i++)
        a[i] = rand() % 100;

    for (size_t i = 0; i < count; i++)
        b[i] = rand() % 100;

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks(
        (COLS + 31) / 32,
        (ROWS + 15) / 16
    );

    double dStart = cpuSecond();

    mulElem<<<numBlocks, threadsPerBlock>>>(
        d_a, d_b, d_c, ROWS, COLS
    );

    cudaError_t err = cudaDeviceSynchronize();

    double dElaps = cpuSecond() - dStart;

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    size_t mismatches = 0;

    for (size_t i = 0; i < count; i++) {
        if (c[i] != a[i] * b[i])
            mismatches++;
    }

    printf("Grid: %u x %u blocks\n", numBlocks.x, numBlocks.y);
    printf("Verification: %zu mismatches out of %zu\n",
           mismatches, count);
    printf("Time taken - %.7f\n", dElaps);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
