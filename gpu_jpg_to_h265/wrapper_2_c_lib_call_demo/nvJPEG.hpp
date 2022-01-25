#ifndef NV_JPEG_
#define NV_JPEG_

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <nvjpeg.h>
#include <cuda.h>
#include <cuda_runtime.h>
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


#define  NVJPEG_MAX_COMPONENT 4
#define  BATCH   8

int dev_malloc(void **p, size_t s);
int dev_free(void *p);

class NvJpeg{
private:
    nvjpegJpegState_t            _nvjpeg_state;
    nvjpegHandle_t               _nvjpeg_handle;
    nvjpegEncoderParams_t        _encode_params;
    nvjpegEncoderState_t         _encoder_state;
public:
    nvjpegOutputFormat_t         _fmt;
    nvjpegImage_t                _imgdesc;
    int                          _width;
    int                          _height;
    int                          _device_id;
    unsigned char *              _pBuffer; 
    int decompress(const unsigned char* jpgBuffer, const uint32_t jpgSize);
    NvJpeg(int, int, int, AVFrame *, nvjpegOutputFormat_t);
    ~NvJpeg();
};

#endif
