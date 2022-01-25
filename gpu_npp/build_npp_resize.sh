#!/bin/sh

g++ npp_resize.cpp -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lnppisu_static -lnpps_static -lnppial_static -lnppist_static -lnppidei_static -lnppif_static -lnppim_static -lnppig_static -lnppicc_static -lnppicom_static -lnppitc_static -lnppc_static  -lculibos -lcudart_static -lpthread -ldl -lrt -I /usr/local/cuda-10.1/include  -L /usr/local/lib -L /usr/local/cuda-10.1/lib64  -o npp_resize
