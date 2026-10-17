#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 512

#define CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while (0)

__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));
    CHECK(cudaMalloc((void**)&d_c, size));

    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    double dStart = cpuSecond();

    add<<<N, 1>>>(d_a, d_b, d_c);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    int mismatches = 0;
    for (int i = 0; i < N; i++)
        if (c[i] != a[i] + b[i])
            mismatches++;

    printf("Verification: %d mismatches out of %d\n", mismatches, N);
    printf("Time taken - %.7f\n", dElaps);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return 0;
}
