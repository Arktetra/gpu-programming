#include<cstdlib>
#include<stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include"includes/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include"includes/stb_image_write.h"

__global__ void convert_rgb_to_gray(unsigned char* rgb_data, unsigned char* gray_data, int width, int height, int channels) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < width && row < height) {
        int idx = width * row + col;
        int offset = idx * channels;

        unsigned char r = rgb_data[offset + 0];
        unsigned char g = rgb_data[offset + 1];
        unsigned char b = rgb_data[offset + 2];

        gray_data[idx] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}

int main() {
    unsigned char* rgb_data = nullptr;
    unsigned char* gray_data = nullptr;

    int width, height, channels;
    rgb_data = stbi_load("./imgs/nebula.png", &width, &height, &channels, 0);
    
    cudaMallocHost(&gray_data, width * height * sizeof(unsigned char));

    unsigned char* rgb_data_d = nullptr;
    unsigned char* gray_data_d = nullptr;
    cudaMalloc(&rgb_data_d, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&gray_data_d, width * height * sizeof(unsigned char));

    cudaMemcpy(rgb_data_d, rgb_data, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(gray_data_d, gray_data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 num_threads_per_block(32, 32);
    dim3 num_blocks((width + 32 - 1) / 32, (height + 32 - 1) / 32);
    convert_rgb_to_gray<<<num_blocks, num_threads_per_block>>>(rgb_data_d, gray_data_d, width, height, channels);

    cudaMemcpy(gray_data, gray_data_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_png("./results/nebula.png", width, height, 1, gray_data, width);
    // printf("%d", gray_data[0]);

    cudaFree(rgb_data_d);
    cudaFree(gray_data_d);
    cudaFreeHost(gray_data);
    stbi_image_free(rgb_data);

    return 0;
}