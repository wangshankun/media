#ifndef WRAPPER_H_
#define WRAPPER_H_


#ifdef __cplusplus
extern "C"
{
#endif

#include <nvjpeg.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavdevice/avdevice.h>
#include <libavutil/hwcontext_cuda.h>

void *call_NvJpeg_Create(int, int, int, AVFrame*, nvjpegOutputFormat_t);
void call_NvJpeg_Destroy(void *p);
int  call_NvJpeg_Decompress(void *p, const unsigned char* jpgBuffer, const uint32_t jpgSize);

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H_
