#include <iostream>
#include <string>
#include "nvJPEG.hpp"
#include "nvJPEG_helper.hpp"
#include<time.h>

int main()
{
    #define WIDTH   1920
    #define HIEGHT  1080
    #define CHANNEL 3 
    #define BATCH   8
    
    FileNames           file_list;
    FileData            file_data(BATCH);
    std::vector<size_t> file_len(BATCH);
    GPUData             gpu_rgb_bufs(BATCH);
    std::vector<nvjpegImage_t> iouts(BATCH);//这个vector是直接初始化8个成员，其它是一个个push进去8个成员
    
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");
    file_list.push_back("0001.jpg");

    for (int i = 0; i < BATCH; i++)
    {
        std::ifstream input(file_list[i], std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
          std::cerr << "Cannot open image: ";
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (file_data[i].size() < file_size)
        {
            file_data[i].resize(file_size);
        }

        if (!input.read(file_data[i].data(), file_size)) 
        {
            std::cerr << "Cannot read from file: ";
        }
        
        file_len[i] = file_size;
        checkCudaErrors(cudaMalloc(&(gpu_rgb_bufs[i]), WIDTH * HIEGHT * CHANNEL));
        
        iouts[i].pitch[0] = WIDTH * CHANNEL;
        checkCudaErrors(cudaMalloc(&(iouts[i].channel[0]), WIDTH * HIEGHT * CHANNEL));
    }
 
    NvJpeg jpeg_dec(BATCH, CHANNEL, HIEGHT, WIDTH, 0, NVJPEG_OUTPUT_RGBI); 
    //warm up
    jpeg_dec.decompress_to_gpubuf_batch(file_data, file_len, iouts, gpu_rgb_bufs);

    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    jpeg_dec.decompress_to_gpubuf_batch(file_data, file_len, iouts, gpu_rgb_bufs);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("elapsed time:%f\r\n",elapsed);

    return EXIT_SUCCESS;
}
