#!/bin/sh
g++ -std=c++11 -I/usr/local/cuda-10.1/include -I../../NvCodec -I../../NvCodec/NvDecoder -I../../NvCodec/NvEncoder -I../../NvCodec/Common -I../../../include -I/usr/local//include   -o NvDecoder.o -c ../../NvCodec/NvDecoder/NvDecoder.cpp

g++ -std=c++11 -I${OPENCV_PATH}/include -I/usr/local/cuda-10.1/include -I../../NvCodec -I../../NvCodec/NvDecoder -I../../NvCodec/NvEncoder -I../../NvCodec/Common -I../../../include -I/usr/local//include   -o AppDec.o -c AppDec.cpp

g++ -std=c++11 -o AppDec AppDec.o NvDecoder.o  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64 -L/usr/local/cuda-10.1/lib64/stubs -L../../../Lib/linux/stubs/x86_64 -ldl -lcuda -lnvcuvid -L/usr/local//lib -lavcodec -lavutil -lavformat -lnppisu_static -lnpps_static -lnppial_static -lnppist_static -lnppidei_static -lnppif_static -lnppim_static -lnppig_static -lnppicc_static -lnppicom_static -lnppitc_static -lnppc_static  -lculibos -lcudart_static -lpthread -lrt -lnvjpeg

./AppDec -i test.hevc -o v.yuv

