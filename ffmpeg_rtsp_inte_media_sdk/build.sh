#!/bin/sh
g++ -std=c++11 -w -g qsvdec.cpp -o test  -fpermissive \
 -I/home/mis5032/work/ffmpeg-4.2.1/install/include \
 -I/opt/intel/mediasdk/include \
 -I/opt/intel/openvino/opencv/include \
 -L/home/mis5032/work/ffmpeg-4.2.1/install/lib \
 -L/opt/intel/mediasdk/lib64 \
 -L/opt/intel/openvino/opencv/lib \
 -lopencv_core  -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui  \
 -ldl -lpthread\
 -lavformat -lavdevice -lavcodec -lavutil -lavfilter -lswscale -lswresample \
 -lmfxhw64 -lmfx \
 -Wl,-rpath,/opt/intel/openvino/opencv/lib:/opt/intel/mediasdk/lib64:/home/mis5032/work/ffmpeg-4.2.1/install/lib
