#!/bin/sh

/usr/bin/c++ -g -w -O3 -I/usr/local/cuda-10.0/include -I/usr/local/include -I/home/shankun.shankunwan/opencv_compile/install/include/opencv4  -Wall -DNDEBUG -fPIE  -std=gnu++11  -c cu_cpu.cpp

/usr/bin/c++ -g -w -O3 -Wall -rdynamic cu_cpu.o -o cu_cpu -L/usr/local/lib -L/usr/local/cuda-10.0/lib64/stubs  -L/usr/local/cuda-10.0/lib64 -L/home/shankun.shankunwan/opencv_compile/install/lib64  -Wl,-rpath,/usr/local/lib:/usr/local/cuda-10.0/lib64:/home/shankun.shankunwan/opencv_compile/install/lib64 -lpthread  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs -lopencv_cudaimgproc  -lopencv_cudacodec -lopencv_cudaarithm -lopencv_cudawarping 


rm -rf *.o
