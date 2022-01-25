#!/bin/sh

g++ npp_batch_re.cpp -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lfreeimage -lnppisu_static -lnpps_static -lnppial_static -lnppist_static -lnppidei_static -lnppif_static -lnppim_static -lnppig_static -lnppicc_static -lnppicom_static -lnppitc_static -lnppc_static  -lculibos -lcudart_static -lpthread -ldl -lrt -I /usr/local/cuda-10.1/include -I /usr/local/cuda-10.1/samples/7_CUDALibraries/common/FreeImage/include -L /usr/local/lib -L /usr/local/cuda-10.1/lib64 -L /usr/local/cuda-10.1/samples/7_CUDALibraries/common/FreeImage/lib/linux/x86_64 -o npp_batch_re
