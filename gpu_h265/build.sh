#!/bin/sh

#g++ -w -O3 hevc_nvenc.cpp -lavutil -lavformat -lavcodec -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64  -o test



g++ -w -O3 hevc_nvenc.cpp -lavutil -lavformat -lavcodec -lavdevice -lavfilter  -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64  -o test_cpu_ram

cuda_path=/usr/local/cuda-10.1
${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -w -g hevc_nvenc_gram.cpp -lavutil -lavformat -lavcodec -lavdevice -lavfilter  -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64  -o test_gpu_ram
