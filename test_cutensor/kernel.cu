#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "cutensor.h"

#include <unordered_map>
#include <vector>

#include <iostream>
#include <iomanip>

using namespace std;

#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

void printMatrix(float* array, int row, int column) {
    int row_offset = row / 4 - 1;
    int column_offset = column / 4 - 1;

    cout << setw(12) << "row/col";

    for (int i = 0; i < column; i += column_offset) {
        cout << setw(12) << i;
        if (i + column_offset < column) {
            cout << setw(12) << "...";
        }
    }

    cout << "\n\n";

    for (int i = 0; i < row; i += row_offset) {
        cout << setw(12) << i << ")";
        for (int j = 0; j < column; j += column_offset) {
            cout << fixed << setprecision(1) << setw(12) << array[i + (j * row)];
            if (j + column_offset < column) {
                cout << setw(12) << "...";
            }
        }
        cout << "\n";
        if (i + row_offset < row) {
            cout << setw(12) << ".\n";
            cout << setw(12) << ".\n";
            cout << setw(12) << ".\n";
        }
    }
    cout << endl;
}

int main(int argc, char** argv)
{
    srand(time(0));

    // CUDA types
    cudaDataType_t type_big = CUDA_R_32F;
    cudaDataType_t type_core = CUDA_R_32F;
    cudaDataType_t type_data_result = CUDA_R_32F;
    cutensorComputeType_t type_compute_result = CUTENSOR_COMPUTE_32F;

    float alpha = (float)1;
    float beta = (float)0;

    std::vector<int> mode_big{ 'm','k' };
    std::vector<int> mode_core{ 'k', 'n' };
    std::vector<int> mode_result{ 'm','n' };
    int nmodeA = mode_big.size();
    int nmodeB = mode_core.size();
    int nmodeC = mode_result.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 512;
    extent['k'] = 128;
    extent['n'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extent_result;
    for (auto mode : mode_result)
        extent_result.push_back(extent[mode]);

    std::vector<int64_t> extent_big;
    for (auto mode : mode_big)
        extent_big.push_back(extent[mode]);

    std::vector<int64_t> extent_core;
    for (auto mode : mode_core)
        extent_core.push_back(extent[mode]);

    // Number of elements of each tensor
    size_t elements_big = 1;
    for (auto mode : mode_big)
        elements_big *= extent[mode];
    size_t elements_core = 1;
    for (auto mode : mode_core)
        elements_core *= extent[mode];
    size_t elements_result = 1;
    for (auto mode : mode_result)
        elements_result *= extent[mode];

    size_t size_big = sizeof(float) * elements_big;
    size_t size_core = sizeof(float) * elements_core;
    size_t size_result = sizeof(float) * elements_result;

    void* big_device, * code_device, * result_device;
    cudaMalloc((void**)&big_device, size_big);
    cudaMalloc((void**)&code_device, size_core);
    cudaMalloc((void**)&result_device, size_result);

    float* big_host = (float*)malloc(sizeof(float) * elements_big);
    float* core_host = (float*)malloc(sizeof(float) * elements_core);
    float* result_host = (float*)malloc(sizeof(float) * elements_result);
    
    for (int64_t i = 0; i < elements_big; i++) {
        big_host[i] = (float)(rand() % 1000);
    }
        
    for (int64_t i = 0; i < elements_core; i++)
        core_host[i] = (float)(rand() % 1000);
    
    for (int64_t i = 0; i < elements_result; i++)
        result_host[i] = 0;
        
        
    std::cout << "\nA:\n";

    printMatrix(big_host, extent['m'], extent['k']);
    std::cout << "\n";

    std::cout << "\nB:\n";

    printMatrix(core_host, extent['k'], extent['n']);
    std::cout << "\n";

    
    std::cout << "\nC:\n";

    printMatrix(result_host, extent['m'], extent['n']);
    std::cout << "\n";
    
    cudaMemcpy(result_device, result_host, size_result, cudaMemcpyHostToDevice);
    cudaMemcpy(big_device, big_host, size_big, cudaMemcpyHostToDevice);
    cudaMemcpy(code_device, core_host, size_core, cudaMemcpyHostToDevice);
    
    /* ***************************** */

    cutensorHandle_t handle;
    cutensorInit(&handle);

    cutensorTensorDescriptor_t desc_big;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
        &desc_big,
        nmodeA,
        extent_big.data(),
        NULL,
        type_big, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t desc_core;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
        &desc_core,
        nmodeB,
        extent_core.data(),
        NULL,
        type_core, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t desc_result;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
        &desc_result,
        nmodeC,
        extent_result.data(),
        NULL,//stride
        type_data_result, CUTENSOR_OP_IDENTITY));

    /* ***************************** */

    uint32_t alignment_requirement_big;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
        big_device,
        &desc_big,
        &alignment_requirement_big));

    uint32_t alignment_requirement_core;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
        code_device,
        &desc_core,
        &alignment_requirement_core));

    uint32_t alignment_requirement_result;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
        result_device,
        &desc_result,
        &alignment_requirement_result));

    /* ***************************** */

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(&handle,
        &desc,
        &desc_big, mode_big.data(), alignment_requirement_big,
        &desc_core, mode_core.data(), alignment_requirement_core,
        &desc_result, mode_result.data(), alignment_requirement_result,
        &desc_result, mode_result.data(), alignment_requirement_result,
        type_compute_result));

    /* ***************************** */

    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind(
        &handle, &find,
        CUTENSOR_ALGO_GETT));

    /* ***************************** */

    size_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspace(&handle,
        &desc,
        &find,
        CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void* work = nullptr;
    if (worksize > 0) {
        if (cudaSuccess != cudaMalloc(&work, worksize)) {
            work = nullptr;
            worksize = 0;
        }
    }

    /* ***************************** */

    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(&handle,
        &plan,
        &desc,
        &find,
        worksize));

    /* ***************************** */

    cutensorStatus_t err;

    err = cutensorContraction(&handle,
        &plan,
        (void*)&alpha, big_device,
        code_device,
        (void*)&beta, result_device,
        result_device,
        work, worksize, 0 /* stream */);
    cudaDeviceSynchronize();

    std::cout << "\n" << work << "\n";

    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        printf("ERROR: %s\n", cutensorGetErrorString(err));
    }

    cudaMemcpy(result_host, result_device, size_result, cudaMemcpyDeviceToHost);

    std::cout << "\n";

    printMatrix(result_host, extent['m'], extent['n']);
    std::cout << "\n";

    /* ***************************** */

    if (big_host) free(big_host);
    if (core_host) free(core_host);
    if (result_host) free(result_host);
    if (big_device) cudaFree(big_device);
    if (code_device) cudaFree(code_device);
    if (result_device) cudaFree(result_device);
    if (work) cudaFree(work);

    printf("End\n");

    return 0;
}