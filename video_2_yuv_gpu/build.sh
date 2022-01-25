#!/bin/sh
gcc demuxing_decoding.c -o demuxing_decoding -L/usr/local//lib -lavdevice -lavformat -lavfilter -lavcodec -lswresample -lswscale -lavutil
