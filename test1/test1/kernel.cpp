
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <iomanip>


#define ROW_NUM_BIG 512
#define COLUMN_NUM_BIG 128

#define ROW_NUM_CORE 128
#define COLUMN_NUM_CORE 64

#define ROW_NUM_RESULT ROW_NUM_BIG
#define COLUMN_NUM_RESULT COLUMN_NUM_CORE


using namespace std;

void generateRandElement(float* array, int row, int column) {
    srand(time(0));
    for (int i = 0; i < row*column; i++) {
        array[i] = (float)(rand()%1000);
    }
}

void printMatrix(float* array, int row, int column) {
    int row_offset = row / 4 - 1;
    int column_offset = column / 4 - 1;

    cout << setw(12) << "row/col";

    for (int i = 0; i < column; i += column_offset) {
        cout << setw(12) << i;
        if (i+ column_offset < column) {
            cout << setw(12) << "...";
        }
    }

    cout << "\n\n";

    for (int i = 0; i < row; i+=row_offset) {
        cout << setw(12) << i << ")";
        for (int j = 0; j < column; j+= column_offset) {
            cout << fixed << setprecision(1) << setw(12) << array[i + (j * row)];
            if (j+ column_offset < column) {
                cout << setw(12) << "...";
            }
        }
        cout << "\n";
        if (i+ row_offset < row) {
            cout << setw(12) << ".\n";
            cout << setw(12) << ".\n";
            cout << setw(12) << ".\n";
        }
    }
    cout << endl;
}

void checkCUBLASError(int status) {
    if (status != 0) {
        cout << "\nCuBLAS error status: " << status << "\n";
        exit(EXIT_FAILURE);
    }
}


int main()
{
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float* big_array = (float*)malloc((COLUMN_NUM_BIG * ROW_NUM_BIG) * sizeof(float));
    printf("row number result: %i\n", ROW_NUM_BIG);
    printf("column number result: %i\n\n", COLUMN_NUM_BIG);

    float* core_array = (float*)malloc(COLUMN_NUM_CORE * ROW_NUM_CORE * sizeof(float));
    printf("row number result: %i\n", ROW_NUM_CORE);
    printf("column number result: %i\n\n", COLUMN_NUM_CORE);

    float* result_array = (float*)malloc(COLUMN_NUM_RESULT * ROW_NUM_RESULT * sizeof(float));
    printf("row number result: %i\n", ROW_NUM_RESULT);
    printf("column number result: %i\n\n", COLUMN_NUM_RESULT);



    generateRandElement(big_array, ROW_NUM_BIG, COLUMN_NUM_BIG);
    
    cout << "big_array:\n\n";
    printMatrix(big_array, ROW_NUM_BIG, COLUMN_NUM_BIG);
    cout << "\n\n";

    generateRandElement(core_array, ROW_NUM_CORE, COLUMN_NUM_CORE);
    
    cout << "core_array:\n\n";
    printMatrix(core_array, ROW_NUM_CORE, COLUMN_NUM_CORE);
    cout << "\n\n";

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    checkCUBLASError(status);

    float* device_big, * device_core, * device_result;

    cudaMalloc(
        (void**)&device_big,
        ROW_NUM_BIG * COLUMN_NUM_BIG * sizeof(float)
    );
    cudaMalloc(
        (void**)&device_core,
        ROW_NUM_CORE * COLUMN_NUM_CORE * sizeof(float)
    );
    cudaMalloc(
        (void**)&device_result,
        ROW_NUM_RESULT * COLUMN_NUM_RESULT * sizeof(float)
    );

    status = cublasSetVector(
        ROW_NUM_BIG * COLUMN_NUM_BIG, // Количество элементов, сохраняемых в видеопамяти
        sizeof(float), // размер каждого элемента
        big_array, // Начальный адрес хоста
        1, // Интервал хранения между последовательными элементами
        device_big, // начальный адрес GPU
        1 // Интервал хранения между последовательными элементами
    );
    checkCUBLASError(status);

    status = cublasSetVector(
        ROW_NUM_CORE * COLUMN_NUM_CORE,
        sizeof(float),
        core_array,
        1,
        device_core,
        1
    );
    checkCUBLASError(status);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != 0) {
        cout << "\nerror cudaThreadSynchronize: " << error << "\n";
        exit(EXIT_FAILURE);
    }

    float alpha = 1; float beta = 0;
    status = cublasSgemm(
        handle, // объект библиотеки blas 
        CUBLAS_OP_N, // Матрица параметров атрибута
        CUBLAS_OP_N, // Параметры атрибута матрицы B 
        ROW_NUM_BIG, // A, C строки 
        COLUMN_NUM_CORE, // B, C
        COLUMN_NUM_BIG, // количество столбцов в A и количество строк в B
        &alpha, // Альфа-значение выражения
        device_big, // адрес в видеопамяти
        ROW_NUM_BIG,    // lda
        device_core, // B в видеопамяти
        COLUMN_NUM_BIG,    // ldb
        &beta, // β значение выражения
        device_result, // адрес C в видеопамяти (матрица результатов)
        ROW_NUM_RESULT    // ldc
    );
    checkCUBLASError(status);

    error = cudaDeviceSynchronize();
    if (error != 0) {
        cout << "\nerror cudaThreadSynchronize: " << error << "\n";
        exit(EXIT_FAILURE);
    }

    status = cublasGetVector(
        COLUMN_NUM_RESULT * ROW_NUM_RESULT, // Количество элементов, которые нужно извлечь
        sizeof(float), // размер каждого элемента
        device_result, // начальный адрес GPU
        1, // Интервал хранения между последовательными элементами
        result_array, // Начальный адрес хоста
        1 // Интервал хранения между последовательными элементами
    );
    checkCUBLASError(status);

    cout << "result:\n\n";
    printMatrix(result_array, ROW_NUM_RESULT, COLUMN_NUM_RESULT);
    cout << "\n\n";

    free(big_array);
    free(core_array);
    free(result_array);
    cudaFree(device_big);
    cudaFree(device_core);
    cudaFree(device_result);

    cublasDestroy(handle);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing by the GPU: %.2f milliseconds\n", gpuTime);

    cout << "End\n";

    return 0;
}
