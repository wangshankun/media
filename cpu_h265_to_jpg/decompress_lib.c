#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <math.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>


#include "turbojpeg.h"

#define MAX_JPG_BUF_SIZE    4  * 1024 * 1024
#define MAX_VIDEO_BUF_SIZE  32 * 1024 * 1024


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


int tyuv2jpeg(unsigned char* yuv_buffer, int yuv_size, int width, int height, int subsample, 
              unsigned char** jpeg_buffer, unsigned long* jpeg_size, int quality)
{
    tjhandle handle = NULL;
    int flags = 0;
    int padding = 1; // 1或4均可，但不能是0
    int need_size = 0;
    int ret = 0;
 
    handle = tjInitCompress();
   
    flags |= 0;
 
    need_size = tjBufSizeYUV2(width, padding, height, subsample);
    if (need_size != yuv_size)
    {
        printf("we detect yuv size: %d, but you give: %d, check again.\n", need_size, yuv_size);
        return 0;
    }
 
    ret = tjCompressFromYUV(handle, yuv_buffer, width, padding, height, subsample, jpeg_buffer, jpeg_size, quality, flags);
    if (ret < 0)
    {
        printf("compress to jpeg failed: %s\n", tjGetErrorStr());
    }
 
    tjDestroy(handle);
 
    return ret;
}

struct buffer_data 
{
    uint8_t *ptr;
    size_t size; ///< size left in the buffer
};

static int read_packet(void *opaque, uint8_t *buf, int buf_size)
{
    struct buffer_data *bd = (struct buffer_data *)opaque;
    buf_size = FFMIN(buf_size, bd->size);

    if (!buf_size)
        return AVERROR_EOF;
    //printf("ptr:%p size:%zu\n", bd->ptr, bd->size);

    /* copy internal buffer data to buf */
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr  += buf_size;
    bd->size -= buf_size;

    return buf_size;
}
int main()
{
    //文件格式上下文
    AVFormatContext    *pFormatCtx;
    int        i = 0,  videoindex;
    AVCodecContext     *pCodecCtx;
    AVCodec            *pCodec;
    AVFrame   *pFrame, *pFrameYUV;
    unsigned char *out_buffer;
    AVPacket *packet;
 
    int y_size;
    int ret, got_picture;
    struct SwsContext *img_convert_ctx;

    av_register_all();
    avformat_network_init();
    pFormatCtx = avformat_alloc_context();
    
    //读取h265视频到内存中
    unsigned char* video_buffer = ( unsigned char*)malloc(16793612);
    readfile("test.hevc", video_buffer, 16793612);

    struct buffer_data bd = { 0 };
    bd.ptr                = video_buffer;
    bd.size               = 16793612;

    size_t avio_ctx_buffer_size = 40960;//申请avio
    uint8_t* avio_ctx_buffer    = av_malloc(avio_ctx_buffer_size);
    if (!avio_ctx_buffer) 
    {
        ret = AVERROR(ENOMEM);
        return ret;
    }
    AVIOContext* avio_ctx = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size,
                                               0, &bd, &read_packet, NULL, NULL);
    if (!avio_ctx) 
    {
        ret = AVERROR(ENOMEM);
        return ret;
    }
    pFormatCtx->pb = avio_ctx;

    if (avformat_open_input(&pFormatCtx, NULL, NULL, NULL) != 0) 
    {
        printf("Couldn't open input stream.\n");
        return -1;
    }
    //读取一部分视音频数据并且获得一些相关的信息
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0) 
    {
        printf("Couldn't find stream information.\n");
        return -1;
    }
 
    //查找视频编码索引
    videoindex = -1;
    for (i = 0; i < pFormatCtx->nb_streams; i++)
    {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoindex = i;
            break;
        }
    }
 
    if (videoindex == -1)
    {
        printf("Didn't find a video stream.\n");
        return -1;
    }
 
    //编解码上下文
    pCodecCtx = pFormatCtx->streams[videoindex]->codec;
    //查找解码器
    pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
    if (pCodec == NULL) 
    {
        printf("Codec not found.\n");
        return -1;
    }
    //打开解码器
    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) 
    {
        printf("Could not open codec.\n");
        return -1;
    }
 
    int align = 32;//1是不对齐
    int yuv_buf_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height, align);
    //申请AVFrame，用于原始视频
    pFrame = av_frame_alloc();
    //申请AVFrame，用于yuv视频
    pFrameYUV = av_frame_alloc();
    //分配内存，用于图像格式转换
    out_buffer = (unsigned char *)av_malloc(yuv_buf_size);
    av_image_fill_arrays(pFrameYUV->data, pFrameYUV->linesize, out_buffer, AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height, align);
    packet = (AVPacket *)av_malloc(sizeof(AVPacket));

    //申请转换上下文
    img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,
    pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
 
    int count = 0;
    unsigned char* out_jpg = (unsigned char*)malloc(MAX_JPG_BUF_SIZE);
    unsigned long jpg_size = 0;
    char jpg_name[64]      = {0};
    
    //读取数据
    while (av_read_frame(pFormatCtx, packet) >= 0) 
    {
        if (packet->stream_index == videoindex) 
        {
            ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
            if (ret < 0)
            {
                printf("Decode Error.\n");
                return -1;
            }
 
            if (got_picture >= 1) 
            {
                //成功解码一帧
                sws_scale(img_convert_ctx, (const unsigned char* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
                    pFrameYUV->data, pFrameYUV->linesize);//转换图像格式

                tyuv2jpeg(pFrameYUV->data[0], yuv_buf_size, pCodecCtx->width, pCodecCtx->height, TJSAMP_420, &out_jpg, &jpg_size, 70);
                count++;
                sprintf(jpg_name, "h265_yuv_%d.jpg", count);
                savefile(jpg_name, out_jpg, jpg_size);

            }
            else
            {
                //未解码到一帧，可能时结尾B帧或延迟帧，在后面做flush decoder处理
            }
        }
        av_free_packet(packet);
    }
 
    //flush decoder
    //FIX: Flush Frames remained in Codec
    while (1) 
    {
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
        if (ret < 0)
        {
            break;
        }
        if (!got_picture)
        {
            break;
        }
 
        sws_scale(img_convert_ctx, (const unsigned char* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
            pFrameYUV->data, pFrameYUV->linesize);

        tyuv2jpeg(pFrameYUV->data[0], yuv_buf_size, pCodecCtx->width, pCodecCtx->height, TJSAMP_420, &out_jpg, &jpg_size, 70);
        count++;
        sprintf(jpg_name, "h265_yuv_%d.jpg", count);
        savefile(jpg_name, out_jpg, jpg_size);
    }
 
    printf("count %d \r\n",count);
    
    sws_freeContext(img_convert_ctx);
    av_frame_free(&pFrameYUV);
    av_frame_free(&pFrame);
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);

    return 0;
}