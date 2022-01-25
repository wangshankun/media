#!/bin/sh
gcc -w -O3 -I/opt/libjpeg-turbo/include/ decompress_lib.c -o test   -lturbojpeg -lavutil -lavformat -lavcodec -lavdevice -lavfilter -lswscale -lpthread -ldl -lrt -L /usr/local/lib -L/opt/libjpeg-turbo/lib64/  -Wl,-rpath,./:/opt/libjpeg-turbo/lib64


