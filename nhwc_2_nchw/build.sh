#!/bin/sh

g++ nhwc_2_nchw.cpp -O3 -w  -L/usr/local/lib  -o nhwc_2_nchw  -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

nvcc -std=c++11 -O3 -I cudnn/include -L cudnn/lib64 -L/usr/local/lib nhwc_2_nchw.cu -o nhwc_2_nchw -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

