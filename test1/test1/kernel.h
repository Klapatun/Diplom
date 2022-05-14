#pragma once

int main()
{
    int c;
    int* dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    add << <1, 1 >> >(1, 2, );

    printf("Hello world!\n");
    return 0;
}
