#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define NX 500
#define NY 500
#define NZ 500

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void multiply3D(
    const int *a, const int *b, int *c,
    int nx, int ny, int nz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        size_t index = ((size_t)z * ny + y) * nx + x;
        c[index] = a[index] * b[index];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
    size_t count = (size_t)NX * NY * NZ;
    size_t size = count * sizeof(int);

    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    if (!a || !b || !c) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < count; i++)
        a[i] = rand() % 100;

    for (size_t i = 0; i < count; i++)
        b[i] = rand() % 100;

    int *d_a, *d_b, *d_c;

    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));
    CHECK(cudaMalloc((void**)&d_c, size));

    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 8);

    dim3 numBlocks(
        (NX + 7) / 8,
        (NY + 7) / 8,
        (NZ + 7) / 8
    );

    double dStart = cpuSecond();

    multiply3D<<<numBlocks, threadsPerBlock>>>(
        d_a, d_b, d_c, NX, NY, NZ
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    size_t mismatches = 0;

    for (size_t i = 0; i < count; i++) {
        if (c[i] != a[i] * b[i])
            mismatches++;
    }

    printf("3D dimensions: %d x %d x %d\n", NX, NY, NZ);
    printf("Grid: %u x %u x %u blocks\n",
           numBlocks.x, numBlocks.y, numBlocks.z);
    printf("Threads per block: 8 x 8 x 8\n");

    printf("\nLast 1000 elements:\n");

    for (size_t i = count - 1000; i < count; i++) {
        size_t z = i / ((size_t)NX * NY);
        size_t y = (i / NX) % NY;
        size_t x = i % NX;

        printf("C[%zu][%zu][%zu] = %d x %d = %d\n",
               z, y, x, a[i], b[i], c[i]);
    }

    printf("\nVerification: %zu mismatches out of %zu\n",
           mismatches, count);
    printf("Time taken - %.7f\n", dElaps);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return 0;
}
