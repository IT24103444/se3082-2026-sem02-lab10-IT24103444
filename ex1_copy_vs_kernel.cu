#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 100000000
#define THREADS_PER_BLOCK 512

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

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
    for (int i = 0; i < N; i++) a[i] = rand() % 100;
    for (int i = 0; i < N; i++) b[i] = rand() % 100;

    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));
    CHECK(cudaMalloc((void**)&d_c, size));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Whole sequence starts before copying inputs
    double totalStart = cpuSecond();

    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Kernel-only timer
    double kernelStart = cpuSecond();

    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double kernelTime = cpuSecond() - kernelStart;

    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    double totalTime = cpuSecond() - totalStart;

    size_t mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] * b[i])
            mismatches++;
    }

    printf("Verification: %zu mismatches out of %d\n", mismatches, N);
    printf("Kernel-only time: %.7f seconds\n", kernelTime);
    printf("Whole-sequence time: %.7f seconds\n", totalTime);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return 0;
}
