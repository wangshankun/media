#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <time.h>
#include "nvJPEG.hpp"
#include "nvJPEG_helper.hpp"

extern "C"
{
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavdevice/avdevice.h>
#include <libavutil/hwcontext_cuda.h>
};

typedef  std::vector<char>  FileData;

static void video_encode_example(const char *filename)
{
    
        //将jpeg图片做二进制文件读入内存;
        FileData raw_data;
        std::ifstream input("0001.jpg",std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
        std::cerr << "Cannot open image: ";
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data.size() < file_size)
        {
            raw_data.resize(file_size);
        }

        if (!input.read(raw_data.data(), file_size))
        {
            std::cerr << "Cannot read from file: ";
        }

        NvJpeg jpeg_dec(1920, 1080, 0, NULL, NVJPEG_OUTPUT_YUV);

        jpeg_dec.decompress_to_gpubuf((unsigned char *)(&raw_data[0]), file_size);
}

int main(int argc, char **argv)
{

    video_encode_example("test.hevc");

    return 0;
}
