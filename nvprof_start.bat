nvprof --log-file ./cuBLAS.log ./test1/test1/x64/Release/test1.exe
nvprof --log-file ./cuTensor.log  ./test_cutensor/x64/Debug/test_cutensor.exe
nvprof --log-file ./thrust.log  ./test2_thrust/x64/Debug/test2_thrust.exe
