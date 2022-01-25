NvCodec例子修改，h265视频解码成jpg图片过程
用到NvCodec解压h265到nv12
用到nppi把nv12转yuv420p
用到nvjpeg把yuv420p转jpeg

在T4上整个过程解压1080p h265视频到jpeg 318fps 耗时0.45s

cd Video\_Codec\_SDK\_9.0.20/Samples/AppDecode/AppDec
替换 AppDec.cpp
执行 ./build.sh



原始例子可以这样执行
export PKG\_CONFIG\_PATH=/usr/local/lib/pkgconfig/ && make && ./AppDec -i test.hevc -o v.yuv
