#!/bin/sh
cuda_path=/usr/local/cuda-10.1

#${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -O3 -I${cuda_path}/samples/common/inc  -m64 -o nvJPEG_helper.o -c nvJPEG_helper.cpp

#${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -O3 -I${cuda_path}/samples/common/inc  -I${cuda_path}/targets/x86_64-linux/include -m64 -o nvJPEG.o -c nvJPEG.cpp

#${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -w -O3 -I${cuda_path}/samples/common/inc  -m64 -o main.o -c main.cpp

#${cuda_path}/bin/nvcc -ccbin g++ -std=c++11 -O3 -m64 *.o -o test -lnvjpeg -lavutil -lavformat -lavcodec -lavdevice -lavfilter -lcuda -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64


#gcc -fpermissive  -g -w  main.c -o test -lavutil -lavformat -lavcodec -lavdevice -lavfilter 

g++ -g -w  main.cpp -o test -lavutil -lavformat -lavcodec -lavdevice -lavfilter -lswscale
