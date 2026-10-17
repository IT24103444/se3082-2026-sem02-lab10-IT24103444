#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 512

__global__ void addT(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    srand(0);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    double dStart = cpuSecond();

    addT<<<1, N>>>(d_a, d_b, d_c);
    cudaError_t err = cudaDeviceSynchronize();

    double dElaps = cpuSecond() - dStart;

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i])
            mismatches++;
    }

    printf("Verification: %d mismatches out of %d\n", mismatches, N);
    printf("Time taken - %.7f\n", dElaps);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
