#!/bin/sh
cuda_path=/usr/local/cuda-10.1

${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -O3 -I${cuda_path}/samples/common/inc  -m64 -o nvJPEG_helper.o -c nvJPEG_helper.cpp


${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -w -O3 -I${cuda_path}/samples/common/inc  -m64     -o main.o -c nvJPEG_encoder.cpp

${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -m64 *.o -o test  -lnvjpeg

rm -r *.o

./test
