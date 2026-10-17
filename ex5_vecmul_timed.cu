#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void vecMul(const int *a, const int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        c[i] = a[i] * b[i];
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
    size_t size = (size_t)N * sizeof(int);

    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    if (!a || !b || !c) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(0);

    for (int i = 0; i < N; i++)
        a[i] = rand() % 100;

    for (int i = 0; i < N; i++)
        b[i] = rand() % 100;

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double dStart = cpuSecond();

    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaError_t err = cudaDeviceSynchronize();

    double dElaps = cpuSecond() - dStart;

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    int mismatches = 0;

    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] * b[i])
            mismatches++;
    }

    printf("Blocks: %d, Threads per block: %d\n",
           blocks, THREADS_PER_BLOCK);
    printf("Verification: %d mismatches out of %d\n", mismatches, N);
    printf("Time taken - %.7f\n", dElaps);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
