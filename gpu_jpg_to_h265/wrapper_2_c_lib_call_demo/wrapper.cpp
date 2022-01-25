#include "nvJPEG.hpp"
#include "wrapper.h"

void *call_NvJpeg_Create(int width = 1920, int height = 1080, int device_id = 0, AVFrame * ff_gpu_frame = NULL, nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_YUV)
{
  return new NvJpeg(width, height, device_id, ff_gpu_frame, fmt);
}

void call_NvJpeg_Destroy(void *p) 
{
  static_cast<NvJpeg *>(p)->~NvJpeg();
}

int call_NvJpeg_Decompress(void *p, const unsigned char* jpgBuffer, const uint32_t jpgSize) 
{
  return static_cast<NvJpeg *>(p)->decompress(jpgBuffer, jpgSize);
}

