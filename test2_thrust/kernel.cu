
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <stdio.h>

#define ROW_NUM_BIG 512
#define COLUMN_NUM_BIG 128

#define ROW_NUM_CORE 128
#define COLUMN_NUM_CORE 64

#define ROW_NUM_RESULT ROW_NUM_BIG
#define COLUMN_NUM_RESULT COLUMN_NUM_CORE

#define PRINT_HOST_MATRIX    1
#define PRINT_RESULT_MATRIX  1

void printMatrix(thrust::device_vector<float> array, int row, int column) {

    int step_row = row/4 - 1;
    int step_column = column / 4 - 1;


    for (int i = 0; i < row; i+= step_row) {
        for (int j = 0; j < column; j+= step_column) {
            std::cout << " " << array[i + j];
            if (j+ step_column < column) {
                std::cout << " ... ";
            }
        }
        std::cout << "\n";

        if (i + step_row < row) {
            std::cout << " .\n";
            std::cout << " .\n";
            std::cout << " .\n";
        }
    }
    std::cout << "\n";
}

struct mul
{
    float* A, * B;
    int m, n, r;
    mul(float* _A, float* _B, int _m, int _n, int _r) : A(_A), B(_B), m(_m), n(_n), r(_r) {};
    __host__ __device__
        float operator()(size_t idx) {
        float sum = 0.0f;
        int row = idx / r;
        int col = idx - (row * r); // cheaper modulo
        for (int i = 0; i < m; i++)
            sum += A[col + row * i] * B[col + row * i];
        return sum;
    }
};


__host__ static __inline__ float ramdom_gen()
{
    return ((float)(rand()%100));
}

int main()
{
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::host_vector<float> big_host(ROW_NUM_BIG * COLUMN_NUM_BIG);
    thrust::generate(big_host.begin(), big_host.end(), ramdom_gen);
    thrust::device_vector<float> big_device(ROW_NUM_BIG * COLUMN_NUM_BIG);
    big_device = big_host;

    if (PRINT_HOST_MATRIX) {
        std::cout << "\n\nData:\n\n";
        printMatrix(big_host, ROW_NUM_BIG, COLUMN_NUM_BIG);
    }

    thrust::host_vector<float> core_host(ROW_NUM_CORE * COLUMN_NUM_CORE);
    thrust::generate(core_host.begin(), core_host.end(), ramdom_gen);
    thrust::device_vector<float> core_device(ROW_NUM_CORE * COLUMN_NUM_CORE);
    core_device = core_host;

    if (PRINT_HOST_MATRIX) {
        std::cout << "\n\Other:\n\n";
        printMatrix(core_host, ROW_NUM_CORE, COLUMN_NUM_CORE);
    }

    thrust::host_vector<float> result_host(ROW_NUM_RESULT * COLUMN_NUM_RESULT);
    thrust::device_vector<float> result_device(ROW_NUM_RESULT * COLUMN_NUM_RESULT);


    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(
        ROW_NUM_RESULT * COLUMN_NUM_RESULT),
        result_device.begin(), 
        mul(
            thrust::raw_pointer_cast(big_device.data()), 
            thrust::raw_pointer_cast(core_device.data()), 
            ROW_NUM_CORE,
            ROW_NUM_RESULT,
            COLUMN_NUM_RESULT
        )
    );
    cudaDeviceSynchronize();

    result_host = result_device;
    if (PRINT_RESULT_MATRIX) {
        std::cout << "\n\Result:\n\n";
        printMatrix(result_host, ROW_NUM_RESULT, COLUMN_NUM_RESULT);
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("time spent executing by the GPU: %.2f milliseconds\n", gpuTime);

    std::cout << "End\n";

    return 0;
}