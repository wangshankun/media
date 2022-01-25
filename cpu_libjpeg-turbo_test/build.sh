#!/bin/sh
gcc -g -I/opt/libjpeg-turbo/include/ -L/opt/libjpeg-turbo/lib64/  main.c -o test -lturbojpeg  -Wl,-rpath,/opt/libjpeg-turbo/lib64/

./test
