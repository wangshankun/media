#!/bin/sh
gcc avio.c -o test -L/home/shankun/video_synopsis/3rdparty/ffmpeg-bin-n4.1.4/lib -I/home/shankun/video_synopsis/3rdparty/ffmpeg-bin-n4.1.4/include -lavcodec -lavutil -lavformat
