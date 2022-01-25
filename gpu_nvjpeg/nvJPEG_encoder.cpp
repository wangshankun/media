#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include "nvJPEG.hpp"
#include "nvJPEG_helper.hpp"
#include<time.h>

#define CUDA_FRAME_ALIGNMENT 256.0

bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

nvjpegEncoderParams_t encode_params;
nvjpegHandle_t nvjpeg_handle;
nvjpegJpegState_t jpeg_state;
nvjpegEncoderState_t encoder_state;

nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_YUV;
    
#define  NVJPEG_MAX_COMPONENT 4
#define BATCH   8

int test()
{ 
    unsigned char * pBuffer = NULL;
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


    std::ifstream input(file_list[0], std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open()))
    {
        std::cerr << "Cannot open image: ";
    }
    // Get the size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (file_data[0].size() < file_size)
    {
        file_data[0].resize(file_size);
    }

    if (!input.read(file_data[0].data(), file_size))
    {
        std::cerr << "Cannot read from file: ";
    }

    file_len[0] = file_size;


    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, (const unsigned char*)(file_data[0].data()), file_len[0], &nComponent, &subsampling, widths, heights))
    {
        std::cerr << "Error decoding JPEG header: " << std::endl;
        return 1;
    }
    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, subsampling, NULL));
    
    unsigned int linesize = (int)(widths[0]/CUDA_FRAME_ALIGNMENT + 1) * CUDA_FRAME_ALIGNMENT;

    cudaError_t eCopy = cudaMalloc(&pBuffer, linesize * heights[0] * NVJPEG_MAX_COMPONENT);
    if(cudaSuccess != eCopy) 
    {
        std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(eCopy) << std::endl;
        return 1;
    }

    nvjpegImage_t imgdesc = 
    {
        {
            pBuffer,
            pBuffer + linesize*heights[0],
            pBuffer + linesize*heights[0]*2,
            pBuffer + linesize*heights[0]*3
        },
        {
            (unsigned int)(linesize),
            (unsigned int)(linesize/2),
            (unsigned int)(linesize/2),
            (unsigned int)(linesize/2)

        }
    };
/*
            cudaError_t eCopy = cudaMalloc(&pBuffer, widths[0] * heights[0] * NVJPEG_MAX_COMPONENT);
            if(cudaSuccess != eCopy)
            {
                std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(eCopy) << std::endl;
                return 1;
            }

            nvjpegImage_t imgdesc =
            {
                {
                    pBuffer,
                   pBuffer + widths[0]*heights[0],
                   pBuffer + widths[0]*heights[0]*2,
                   pBuffer + widths[0]*heights[0]*3
                },
                {
                   (unsigned int)(is_interleaved(output_format) ? widths[0] * 3 : widths[0]),
                   (unsigned int)widths[0],
                   (unsigned int)widths[0],
                   (unsigned int)widths[0]
                }
            };
  */ 

    int nReturnCode = 0;
    cudaDeviceSynchronize();

    nReturnCode = nvjpegDecode(nvjpeg_handle, jpeg_state, (const unsigned char*)(file_data[0].data()), file_len[0], output_format, &imgdesc, NULL);
    cudaDeviceSynchronize();
    if(nReturnCode != 0)
    {
        std::cerr << "Error in nvjpegDecode. " <<  nReturnCode <<  std::endl;
        return 1;
    }

    checkCudaErrors(nvjpegEncodeYUV(nvjpeg_handle,
        encoder_state,
        encode_params,
        &imgdesc,
        subsampling,
        widths[0],
        heights[0],
        NULL));


    std::vector<unsigned char> obuffer;
    size_t length;
    checkCudaErrors(nvjpegEncodeRetrieveBitstream(
        nvjpeg_handle,
        encoder_state,
        NULL,
        &length,
        NULL));
    obuffer.resize(length);
    checkCudaErrors(nvjpegEncodeRetrieveBitstream(
        nvjpeg_handle,
        encoder_state,
        obuffer.data(),
        &length,
        NULL));
        
    std::ofstream outputFile("encode_out_yuv.jpg", std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));

    checkCudaErrors(cudaFree(pBuffer));

    return 0;
}

int dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}
int dev_free(void *p)
{
    return (int)cudaFree(p);
}

int main( ) 
{
    int pidx;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));
    checkCudaErrors(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL));
    checkCudaErrors(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL));
    
    // sample input parameters
    checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, 70, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 1, NULL));

    test();

    checkCudaErrors(nvjpegEncoderParamsDestroy(encode_params));
    checkCudaErrors(nvjpegEncoderStateDestroy(encoder_state));
    checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
    checkCudaErrors(nvjpegDestroy(nvjpeg_handle));

    return pidx;
}
