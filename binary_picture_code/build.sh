#!/bin/sh


opencv_path=/home/shankun.shankunwan/disk1/video_structured_analysis/3rdparty/opencv_3.3

export  LD_LIBRARY_PATH=${opencv_path}/lib64
export  CPLUS_INCLUDE_PATH=${opencv_path}/include

g++ -O3  -I${cuda_path}/include \
 -I${opencv_path}/include -L /usr/local/lib \
 mask_png.cpp -o mask_png -std=gnu++11 -lboost_serialization -lboost_system \
 -L${opencv_path}/lib64/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio \
 -Wl,-rpath,${opencv_path}/lib64/
