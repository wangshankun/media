#!/bin/sh

#export PKG_CONFIG_PATH=/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/lib64/pkgconfig

export  LD_LIBRARY_PATH=/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/lib64
export  CPLUS_INCLUDE_PATH=/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/includ

#g++ -std=c++11 -o test main.cpp -I/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/include/ -I/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/include/opnecv -L/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/lib64 -lopencv_core  -lopencv_imgproc  -lopencv_highgui


g++ -O3 -std=c++11 -o test main.cpp `pkg-config opencv --cflags --libs` -Wl,-rpath,/home/shankun/video_synopsis/3rdparty/opencv-3.4.8/lib64:/home/shankun/video_synopsis/3rdparty/ffmpeg-bin-n4.1.4/lib/
