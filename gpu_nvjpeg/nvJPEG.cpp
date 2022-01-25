#include "nvJPEG.hpp" 
#include "nvJPEG_helper.hpp" 

int dev_malloc(void **p, size_t s)
{ 
    return (int)cudaMalloc(p, s); 
}
int dev_free(void *p)
{ 
    return (int)cudaFree(p); 
}

NvJpeg::NvJpeg(int batch = 1, int channel = 3, int width = 1920, int height = 1080, int device_id = 0, nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI)
{

   _batch      = batch;
   _channel    = channel;
   _width      = width;
   _height     = height;
   _device_id  = device_id;
   _fmt        = fmt;

    _dev_allocator = {&dev_malloc, &dev_free};

    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &_dev_allocator, &_nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(_nvjpeg_handle, &_nvjpeg_state));
    checkCudaErrors(nvjpegDecodeBatchedInitialize(_nvjpeg_handle, _nvjpeg_state, _batch, 1, _fmt));
        
    int devID = gpuDeviceInit(_device_id);
    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
    }

    _iout.pitch[0] = _channel * _width;
    checkCudaErrors(cudaMalloc(&_iout.channel[0], _height * _width * _channel));

    _vchanRGB = new std::vector<unsigned char>(_height * _width * _channel);
    checkCudaErrors(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

NvJpeg::~NvJpeg()
{
    checkCudaErrors(cudaStreamDestroy(_stream));
    checkCudaErrors(cudaFree(_iout.channel[0]));
    checkCudaErrors(nvjpegJpegStateDestroy(_nvjpeg_state));
    checkCudaErrors(nvjpegDestroy(_nvjpeg_handle));
    delete _vchanRGB;
}

int NvJpeg::decompress_to_hostbuf(std::vector<char> &jpgBuffer, const uint32_t jpgSize, std::vector<unsigned char> **bmp)
{

    checkCudaErrors(cudaStreamSynchronize(_stream));
    checkCudaErrors(nvjpegDecode(_nvjpeg_handle, _nvjpeg_state,
                                     (const unsigned char *)jpgBuffer.data(),
                                     jpgSize, _fmt, &_iout,
                                     _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));
                         
    unsigned char* chanRGB = (*_vchanRGB).data();
    checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)_width * _channel, _iout.channel[0], (size_t)_iout.pitch[0],
                               _width * _channel, _height, cudaMemcpyDeviceToHost));
    *bmp = _vchanRGB;
    return EXIT_SUCCESS;
}

int NvJpeg::decompress_to_file(std::vector<char> &jpgBuffer, const uint32_t jpgSize, char* filename)
{

    checkCudaErrors(cudaStreamSynchronize(_stream));
    checkCudaErrors(nvjpegDecode(_nvjpeg_handle, _nvjpeg_state,
                                     (const unsigned char *)jpgBuffer.data(),
                                     jpgSize, _fmt, &_iout,
                                     _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));
                         
    unsigned char* chanRGB = (*_vchanRGB).data();
    checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)_width * _channel, _iout.channel[0], (size_t)_iout.pitch[0],
                               _width * _channel, _height, cudaMemcpyDeviceToHost));

    writeBMPi_test(filename, _vchanRGB, (size_t)_iout.pitch[0], _width, _height);

    return EXIT_SUCCESS;
}

int NvJpeg::decompress_to_gpubuf(std::vector<char> &jpgBuffer, const uint32_t jpgSize, unsigned char* gpubuf)
{

    checkCudaErrors(cudaStreamSynchronize(_stream));
    checkCudaErrors(nvjpegDecode(_nvjpeg_handle, _nvjpeg_state,
                                 (const unsigned char *)jpgBuffer.data(),
                                 jpgSize, _fmt, &_iout,
                                 _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));

    checkCudaErrors(cudaMemcpy2D(gpubuf, (size_t)_width * _channel, _iout.channel[0], (size_t)_iout.pitch[0],
                               _width * _channel, _height, cudaMemcpyDeviceToDevice));

    return EXIT_SUCCESS;
}

int NvJpeg::decompress_to_gpubuf_batch(const FileData &img_data, const std::vector<size_t> &img_len, 
                                             std::vector<nvjpegImage_t> &iouts, GPUData &gpu_out_bufs)
{
    checkCudaErrors(cudaStreamSynchronize(_stream));
    std::vector<const unsigned char *> raw_inputs;
    for (int i = 0; i < img_data.size(); i++) {
      raw_inputs.push_back((const unsigned char *)img_data[i].data());
    }

    int thread_idx = 0;
    for (int i = 0; i < raw_inputs.size(); i++)
    {
        checkCudaErrors(nvjpegDecodeBatchedPhaseOne(_nvjpeg_handle, _nvjpeg_state, raw_inputs[i],
                                                    img_len[i], i, thread_idx, _stream));
    }
    checkCudaErrors(nvjpegDecodeBatchedPhaseTwo(_nvjpeg_handle, _nvjpeg_state, _stream));
    checkCudaErrors(nvjpegDecodeBatchedPhaseThree(_nvjpeg_handle, _nvjpeg_state, iouts.data(), _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));
   /* 
    for (int i = 0; i < gpu_out_bufs.size(); i++)
    {
        printf("%p  %p\r\n",gpu_out_bufs[i], iouts[i].channel[0]);
        printf("%d  %d  %d %d \r\n",(size_t)_width * _channel,(size_t)iouts[i].pitch[0], _width * _channel,_height);
        checkCudaErrors(cudaMemcpy2D(gpu_out_bufs[i], (size_t)_width * _channel, iouts[i].channel[0],
                       (size_t)iouts[i].pitch[0],  _width * _channel, _height, cudaMemcpyDeviceToDevice));
    }
*/
/*
    for (int i = 0; i < raw_inputs.size(); i++)
    {
        writeBMPi("test.bmp", iouts[i].channel[0], iouts[i].pitch[0],1920, 1080);
    }
*/
    return EXIT_SUCCESS;
}
