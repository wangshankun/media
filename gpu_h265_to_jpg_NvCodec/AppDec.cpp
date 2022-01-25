#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <nvjpeg.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>
#include "NvDecoder.h"

#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"

using namespace std;

#define CUDA_FRAME_ALIGNMENT 256.0
#define NVJPEG_MAX_COMPONENT 4

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", file, line,
            static_cast<unsigned int>(result), func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


int dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}
int dev_free(void *p)
{
    return (int)cudaFree(p);
}

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


class FFmpegBuf:public FFmpegDemuxer::DataProvider 
{
    public:
        uint8_t* _src_buf;
        int      _max_size;
        int      _has_read_size;
        
        int GetData(uint8_t *pBuf, int nBuf)
        {
            if (_has_read_size + nBuf <= _max_size)
            {
                memcpy(pBuf, _src_buf + _has_read_size, nBuf);
                _has_read_size = _has_read_size + nBuf;
                return nBuf;
            }
            else if (_has_read_size < _max_size)
            {
                int res_size =  _max_size - _has_read_size;
                memcpy(pBuf, _src_buf + _has_read_size, res_size);
                _has_read_size = _max_size;
                return res_size;
            }
            else
            {
                return -1;//EOF标志
            }
        }
        
        FFmpegBuf(uint8_t* src_buf, int max_size)
        {
            _src_buf       = src_buf;
            _max_size      = max_size;
            _has_read_size = 0;
        }
        
        ~FFmpegBuf()
        {
            _max_size = 0;
            free(_src_buf);
        }
};

void DecodeMediaFile(const char *szInFilePath)
{

    av_log_set_level(0);

    int _device_id = 0;
    ck(cuInit(_device_id));
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, _device_id));

    nvjpegJpegState_t            _nvjpeg_state;
    nvjpegHandle_t               _nvjpeg_handle;
    nvjpegEncoderParams_t        _encode_params;
    nvjpegEncoderState_t         _encoder_state;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &_nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(_nvjpeg_handle, &_nvjpeg_state));
    checkCudaErrors(nvjpegEncoderStateCreate(_nvjpeg_handle, &_encoder_state, NULL));
    checkCudaErrors(nvjpegEncoderParamsCreate(_nvjpeg_handle, &_encode_params, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetQuality(_encode_params, 70, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(_encode_params, 1, NULL));

    nvjpegImage_t                _imgdesc;
    unsigned char *              _pBuffer;
    int                          _height = 1080;
    int                          _width  = 1920;

    unsigned int linesize = (int)(_width/CUDA_FRAME_ALIGNMENT + 1) * CUDA_FRAME_ALIGNMENT;
    cudaError_t eCopy = cudaMalloc(&_pBuffer, linesize * _height * NVJPEG_MAX_COMPONENT);
    if(cudaSuccess != eCopy)
    {
        std::cerr << "cudaMalloc failed" << cudaGetErrorString(eCopy) << std::endl;
    }

    _imgdesc.channel[0] = _pBuffer;
    _imgdesc.channel[1] = _pBuffer + linesize * _height;
    _imgdesc.channel[2] = _pBuffer + linesize * _height * 2;
    _imgdesc.channel[3] = _pBuffer + linesize * _height * 3;
    _imgdesc.pitch[0]   = (unsigned int)(linesize);
    _imgdesc.pitch[1]   = (unsigned int)(linesize/2);
    _imgdesc.pitch[2]   = (unsigned int)(linesize/2);
    _imgdesc.pitch[3]   = (unsigned int)(linesize/2);

    NppiSize roi_size       = { _width,  _height };
    Npp8u *pYuv420pDst[3]   = {_imgdesc.channel[0],      _imgdesc.channel[1],      _imgdesc.channel[2]};
    int   pYuv420pPitch[3]  = {(int)(_imgdesc.pitch[0]), (int)(_imgdesc.pitch[1]), (int)(_imgdesc.pitch[2])};


    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL, **ppFrame;
    bool bDecodeOutSemiPlanar = false;

    uint8_t* buf = (uint8_t*)malloc(16793612);
    readfile("test.hevc", buf, 16793612);
    
    FFmpegBuf mem_buf(buf, 16793612);//16793612是视频的大小
    
    FFmpegDemuxer buf_demuxer(&mem_buf);//从文件中test.hevc读取视频流，改为从内存中读(avio)
    NvDecoder dec(cuContext, buf_demuxer.GetWidth(), buf_demuxer.GetHeight(), true, FFmpeg2NvCodecId(buf_demuxer.GetVideoCodec()));
    int count = 0;
    do {
        buf_demuxer.Demux(&pVideo, &nVideoBytes);
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();

        NppStatus result; 
        for (int i = 0; i < nFrameReturned; i++) 
        {
            Npp8u *pNV12Src[2] = {(Npp8u *)(ppFrame[i]),
                                  (Npp8u *)(ppFrame[i]) + (_width * _height) };

            result = nppiNV12ToYUV420_8u_P2P3R(pNV12Src, dec.GetWidth(),
                                               pYuv420pDst, pYuv420pPitch,
                                               roi_size);
            if (result != NPP_SUCCESS)
            {
               std::cerr << "Error executing NV12ToYUV420 -- code: " << result << std::endl;
            }

            nvjpegEncoderParamsSetSamplingFactors(_encode_params, NVJPEG_CSS_420, NULL);

            checkCudaErrors(nvjpegEncodeYUV(_nvjpeg_handle,
                _encoder_state,
                _encode_params,
                &_imgdesc,
                NVJPEG_CSS_420,
                _width,
                _height,
                NULL));

            std::vector<unsigned char> obuffer;
            size_t length;
            checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                _nvjpeg_handle,
                _encoder_state,
                NULL,
                &length,
                NULL));
            obuffer.resize(length);
            checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                _nvjpeg_handle,
                _encoder_state,
                obuffer.data(),
                &length,
                NULL));
      
            count = count + 1;
            printf("count: %d\r\n",count);
            //std::string filename = "h265_yuv_" + std::to_string(count) + ".jpg";
            //std::ofstream outputFile(filename, std::ios::out | std::ios::binary);
            //outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
        }
        nFrame += nFrameReturned;
    } while (nVideoBytes);
}

int main() 
{

    DecodeMediaFile("test.hevc");

    return 0;
}
