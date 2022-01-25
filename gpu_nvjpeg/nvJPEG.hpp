#ifndef NV_JPEG_
#define NV_JPEG_

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <nvjpeg.h>

typedef std::vector<std::string>        FileNames;
typedef std::vector<std::vector<char> > FileData;
typedef std::vector<unsigned char* > GPUData;

int dev_malloc(void **p, size_t s);
int dev_free(void *p);

class NvJpeg{
private:
    nvjpegJpegState_t            _nvjpeg_state;
    nvjpegHandle_t               _nvjpeg_handle;
    nvjpegDevAllocator_t         _dev_allocator;
    cudaStream_t                 _stream;
    nvjpegImage_t                _iout;
    std::vector<unsigned char> * _vchanRGB;
public:
    nvjpegOutputFormat_t         _fmt;
    int                          _channel;
    int                          _width;
    int                          _height;
    int                          _device_id;
    int                          _batch;
    
    int decompress_to_hostbuf(std::vector<char>& jpgBuffer,  const uint32_t jpgSize, std::vector<unsigned char> **bmp);
    int decompress_to_file(std::vector<char>& jpgBuffer,  const uint32_t jpgSize, char* filename);
    int decompress_to_gpubuf(std::vector<char> &jpgBuffer, const uint32_t jpgSize, unsigned char* gpubuf);
    int decompress_to_gpubuf_batch(const FileData &img_data, const std::vector<size_t> &img_len,
                                             std::vector<nvjpegImage_t> &iouts, GPUData &gpu_out_bufs);

    NvJpeg(int, int, int, int, int, nvjpegOutputFormat_t);
    ~NvJpeg();
};

#endif
