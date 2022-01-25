#include "nvJPEG.hpp" 
#include "nvJPEG_helper.hpp" 

#define CUDA_FRAME_ALIGNMENT 256.0

int dev_malloc(void **p, size_t s)
{ 
    return (int)cudaMalloc(p, s); 
}
int dev_free(void *p)
{ 
    return (int)cudaFree(p); 
}

NvJpeg::NvJpeg(int width = 1920, int height = 1080, int device_id = 0, AVFrame * ff_gpu_frame = NULL, nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_YUV)
{

    _width      = width;
    _height     = height;
    _device_id  = device_id;
    _fmt        = fmt;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &_nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(_nvjpeg_handle, &_nvjpeg_state));
    checkCudaErrors(nvjpegEncoderStateCreate(_nvjpeg_handle, &_encoder_state, NULL));
    checkCudaErrors(nvjpegEncoderParamsCreate(_nvjpeg_handle, &_encode_params, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetQuality(_encode_params, 70, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(_encode_params, 1, NULL));

    int devID = gpuDeviceInit(_device_id);
    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
    }
   
    _imgdesc.channel[0] =  ff_gpu_frame->data[0];
    _imgdesc.channel[1] =  ff_gpu_frame->data[1];
    _imgdesc.channel[2] =  ff_gpu_frame->data[2];
    _imgdesc.channel[3] =  ff_gpu_frame->data[3];
    _imgdesc.pitch[0]   = (unsigned int)(ff_gpu_frame->linesize[0]);
    _imgdesc.pitch[1]   = (unsigned int)(ff_gpu_frame->linesize[1]);
    _imgdesc.pitch[2]   = (unsigned int)(ff_gpu_frame->linesize[2]);
    _imgdesc.pitch[3]   = (unsigned int)(ff_gpu_frame->linesize[3]);

}

NvJpeg::~NvJpeg()
{
    checkCudaErrors(nvjpegEncoderParamsDestroy(_encode_params));
    checkCudaErrors(nvjpegEncoderStateDestroy(_encoder_state));
    checkCudaErrors(nvjpegJpegStateDestroy(_nvjpeg_state));
    checkCudaErrors(nvjpegDestroy(_nvjpeg_handle));
}

int NvJpeg::decompress(const unsigned char* jpgBuffer, const uint32_t jpgSize)
{

    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(_nvjpeg_handle, jpgBuffer, jpgSize, &nComponent, &subsampling, widths, heights))
    {
        std::cerr << "Error decoding JPEG header: " << std::endl;
        return 1;
    }
    
    //checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(_encode_params, subsampling, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(_encode_params, NVJPEG_CSS_420, NULL));//??????420

    int nReturnCode = 0;
    cudaDeviceSynchronize();
    nReturnCode = nvjpegDecode(_nvjpeg_handle, _nvjpeg_state, jpgBuffer, jpgSize, _fmt, &_imgdesc, NULL);
    cudaDeviceSynchronize();
    if(nReturnCode != 0)
    {
        std::cerr << "Error in nvjpegDecode. " << nReturnCode << std::endl;
        return 1;
    }
}
