#!/bin/sh

#g++ -w -O3 hevc_nvenc.cpp -lavutil -lavformat -lavcodec -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64  -o test


g++ -w -O3 hevc_nvenc.cpp  -lavutil -lavformat -lavcodec -lavdevice -lavfilter  -lpthread -ldl -lrt -I /usr/local/cuda-10.0/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64  -o test
