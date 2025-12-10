#include <cstdlib>
#include <stdio.h>

__global__ void vec_add(float* x, float* y, float* z, int vector_length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vector_length) {
        z[idx] = x[idx] + y[idx];
    }
}

void vec_add_cpu(float* x, float* y, float* z, int vector_length) {
    for (int i = 0; i < vector_length; i++) {
        z[i] = x[i] + y[i];
    }
}

bool is_close(float* x, float* y, int length, float epsilon = 0.00001) {
    for (int i = 0; i < length; i++) {
        if (fabs(x[i] - y[i]) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, x[i], y[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int vector_length = 1024;
    if (argc >= 2) {
        vector_length = std::atoi(argv[1]);
    }

    // Allocate memory
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* comparison_result = (float*)malloc(vector_length * sizeof(float));

    cudaMallocHost(&x, vector_length * sizeof(float));
    cudaMallocHost(&y, vector_length * sizeof(float));
    cudaMallocHost(&z, vector_length * sizeof(float));

    for (int i = 0; i < vector_length; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    float* x_d = nullptr;
    float* y_d = nullptr;
    float* z_d = nullptr;
    cudaMalloc(&x_d, vector_length * sizeof(float));
    cudaMalloc(&y_d, vector_length * sizeof(float));
    cudaMalloc(&z_d, vector_length * sizeof(float));

    // Copy to the device memory
    cudaMemcpy(x_d, x, vector_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, vector_length * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    unsigned int threads_per_block = 256;
    unsigned int blocks = (vector_length + threads_per_block - 1) / threads_per_block;
    vec_add<<< blocks, threads_per_block >>>(x_d, y_d, z_d, vector_length);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Copy to the host memory
    cudaMemcpy(z, z_d, vector_length * sizeof(float), cudaMemcpyDeviceToHost);

    vec_add_cpu(x, y, comparison_result, vector_length);

    if (is_close(z, comparison_result, vector_length)) {
        printf("CPU and GPU answers match\n");
    } else {
        printf("Error: CPU and GPU answers do not match\n");
    }

    // Deallocate host memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(z);
    free(comparison_result);

    return 0;
}