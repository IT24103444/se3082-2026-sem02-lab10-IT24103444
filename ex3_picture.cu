#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define WIDTH 800
#define HEIGHT 600

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void PictureKernel(
    unsigned char *d_Pin,
    unsigned char *d_Pout,
    int width,
    int height
) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Row < height && Col < width) {
        int index = Row * width + Col;
        int value = 2 * (int)d_Pin[index];

        d_Pout[index] = (value > 255) ? 255 : value;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
    const int N = WIDTH * HEIGHT;
    const size_t size = (size_t)N * sizeof(unsigned char);

    unsigned char *h_Pin = (unsigned char*)malloc(size);
    unsigned char *h_Pout = (unsigned char*)malloc(size);

    if (!h_Pin || !h_Pout) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++)
        h_Pin[i] = rand() % 256;

    unsigned char *d_Pin, *d_Pout;

    CHECK(cudaMalloc((void**)&d_Pin, size));
    CHECK(cudaMalloc((void**)&d_Pout, size));

    CHECK(cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (WIDTH + 15) / 16,
        (HEIGHT + 15) / 16
    );

    double dStart = cpuSecond();

    PictureKernel<<<numBlocks, threadsPerBlock>>>(
        d_Pin, d_Pout, WIDTH, HEIGHT
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    int cappedPixels = 0;

    for (int i = 0; i < N; i++) {
        int expected = 2 * (int)h_Pin[i];

        if (expected > 255) {
            expected = 255;
            cappedPixels++;
        }

        if (h_Pout[i] != expected)
            mismatches++;
    }

    printf("Image size: %d x %d\n", WIDTH, HEIGHT);
    printf("Grid: %u x %u blocks\n", numBlocks.x, numBlocks.y);
    printf("Threads per block: 16 x 16\n");

    printf("\nOriginal pixels (first 50 x 50):\n");

    for (int row = 0; row < 50; row++) {
        for (int col = 0; col < 50; col++)
            printf("%3d ", (int)h_Pin[row * WIDTH + col]);

        printf("\n");
    }

    printf("\nUpdated pixels (first 50 x 50):\n");

    for (int row = 0; row < 50; row++) {
        for (int col = 0; col < 50; col++)
            printf("%3d ", (int)h_Pout[row * WIDTH + col]);

        printf("\n");
    }

    printf("\nVerification: %d mismatches out of %d\n", mismatches, N);
    printf("Capped pixels: %d\n", cappedPixels);
    printf("Time taken - %.7f\n", dElaps);

    CHECK(cudaFree(d_Pin));
    CHECK(cudaFree(d_Pout));

    free(h_Pin);
    free(h_Pout);

    return 0;
}
