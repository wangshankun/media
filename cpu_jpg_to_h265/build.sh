#!/bin/sh

gcc -w -O3 -fPIC -I/opt/libjpeg-turbo/include/ -m64 -o compress_lib.o -c compress_lib.c

gcc -O3 -m64 *.o  -o libcompress.so -shared  -lturbojpeg -lavutil -lavformat -lavcodec -lavdevice -lavfilter -lpthread -ldl -lrt -L /usr/local/lib -L/opt/libjpeg-turbo/lib64/

g++ -std=c++11 -O3 main.cpp -o main -L./ -lcompress -Wl,-rpath,./:/opt/libjpeg-turbo/lib64

rm -r *.o

