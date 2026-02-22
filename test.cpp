#include <iostream>
#include <cstring>
#include <chrono>
#include <omp.h>

#include "conv3d.cpp"

const int volume_depth = 3, volume_height = 50, volume_width = 50;
const int k_depth = 3, k_height = 3, k_width = 3;
float output[volume_depth][volume_height][volume_width];

float apply_kernel(float* volume, float* kernel, 
                  int volume_depth, int volume_height, int volume_width, 
                  int k_depth, int k_height, int k_width,
                  int z, int y, int x,
                int output_depth, int output_height, int output_width) {
    float sum = 0.0f;

    // Apply the kernel to the specified position
    for (int kz = 0; kz < k_depth; kz++) {
        for (int ky = 0; ky < k_height; ky++) {
            for (int kx = 0; kx < k_width; kx++) {
                sum += volume[(z + kz) * volume_height * volume_width + (y + ky) * volume_width + (x + kx)] *
                       kernel[kz * k_height * k_width + ky * k_width + kx];
            }
        }
    }

    return sum;
}

void convolve_3d(float* output, float* volume, int volume_depth, int volume_height, int volume_width, 
                   float* kernel, int k_depth, int k_height, int k_width, 
                   int output_depth, int output_height, int output_width) {

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            output[0 * output_height * output_width + y * output_width + x] =
                apply_kernel(volume, kernel, volume_depth, volume_height, volume_width,
                                k_depth, k_height, k_width, 0, y, x, output_depth, output_height, output_width);
        }
    }
}

void print_res(int output_depth, int output_height, int output_width)
{
    for (int z = 0; z < output_depth; z++) {
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                std::cout << output[z * output_height * output_width + y * output_width + x] << " ";
            }
            std::cout << std::endl;
        }
    }
}

int main() {
    float volume[volume_depth][volume_height][volume_width];
    float* volume_p = &volume[0][0][0];
    for (int i = 0; i < volume_depth * volume_height * volume_width; i++) {
        volume_p[i] = i + 1;
    }
    float kernel[k_depth][k_height][k_width] = {
        {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 0, 0}
        },
        {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}
        }
    };

    float* kernel_p = &kernel[0][0][0];
    int output_depth, output_height, output_width;
    output_depth = volume_depth - k_depth + 1;
    output_height = volume_height - k_height + 1;
    output_width = volume_width - k_width + 1;
    float* output_p = &output[0][0][0];

    std::cout << "---------------------------------------\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    conv3d_3p((void*)(volume_p),(void*)(volume_p + (volume_width * volume_height)),(void*)(volume_p + (2 * volume_width * volume_height)), kernel_p, output, volume_width, volume_height, k_width, k_height, k_depth, 1);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Time taken: " << elapsed.count() << "s\n";
    // print_res(output_depth, output_height, output_width);
    memset(output, 0, sizeof(float) * output_depth * output_height * output_width);
    std::cout << "---------------------------------------\n";
    start_time = std::chrono::high_resolution_clock::now();
    convolve_3d(output_p, volume_p, volume_depth, volume_height, volume_width, kernel_p, k_depth, k_height, k_width, output_depth, output_height, output_width);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    std::cout << "Time taken: " << elapsed.count() << "s\n";
    // print_res(output_depth, output_height, output_width);
    memset(output, 0, sizeof(float) * output_depth * output_height * output_width);
    std::cout << "---------------------------------------\n";

    return 0;
}
