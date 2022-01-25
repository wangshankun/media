#!/bin/sh
cuda_path=/usr/local/cuda-10.1

g++ -std=c++11 -O3 -fPIC -I${cuda_path}/samples/common/inc -I${cuda_path}/targets/x86_64-linux/include -m64 -o nvJPEG_helper.o -c nvJPEG_helper.cpp

g++ -std=c++11 -O3 -fPIC -I${cuda_path}/samples/common/inc -I${cuda_path}/targets/x86_64-linux/include -m64 -o nvJPEG.o -c nvJPEG.cpp

g++ -w -std=c++11 -O3 -fPIC -I${cuda_path}/samples/common/inc -I${cuda_path}/targets/x86_64-linux/include -m64 -o wrapper.o -c wrapper.cpp

gcc -w -O3 -fPIC -I${cuda_path}/targets/x86_64-linux/include  -m64 -o compress_lib.o -c compress_lib.c

gcc -O3 -m64 *.o  -o libcompress.so -shared  -lnvjpeg -lavutil -lavformat -lavcodec -lavdevice -lavfilter -lcuda -lcudart -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64

g++ -std=c++11 -O3 main.cpp -o main -L./ -lcompress -Wl,-rpath,./:/usr/local/cuda-10.1/lib64

rm -r *.o

