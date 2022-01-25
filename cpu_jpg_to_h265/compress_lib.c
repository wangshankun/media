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

#include "turbojpeg.h"

#define MAX_VIDEO_BUF_SIZE  32* 1024*1024


int compress_2_h265_cpu(unsigned char* in_datas, int* sizes, int in_num, unsigned char** out_data, int *out_size)
{
    int ret, i, got_output;

    avcodec_register_all();
    avcodec_register_all();
    avdevice_register_all();
    avfilter_register_all();
    av_register_all();
    avformat_network_init();

    AVCodec *codec;
    AVCodecContext *c = NULL;
    AVFrame *frame;
    AVPacket pkt;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    //av_log_set_level(64);
    
    codec = avcodec_find_encoder(AV_CODEC_ID_H265);
    //codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    /* put sample parameters */
    c->bit_rate = 400000;
    /* resolution must be a multiple of two */
    c->width  = 1920;
    c->height = 1080;
    /* frames per second */
    c->time_base.num = 1;
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    c->max_b_frames = 0;

    AVDictionary *param = 0;
    //H.264
    if (codec->id == AV_CODEC_ID_H264) {
        av_dict_set(&param, "preset", "medium", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
    }
    //H.265
    if (codec->id == AV_CODEC_ID_H265 || codec->id == AV_CODEC_ID_HEVC){
        //av_dict_set(&param, "x265-params", "qp=20", 0);
        av_dict_set(&param, "x265-params", "crf=32", 0);
        av_dict_set(&param, "x265-params", "log-level=-1", 0);//log-level:default is 2(info)
        av_dict_set(&param, "preset", "fast", 0);
        //av_dict_set(&param, "tune", "zero-latency", 0);
    }

    /* open it */
    if (avcodec_open2(c, codec, &param) < 0) {
        fprintf(stderr, "Could not open codec\n");
        system("pause");
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;

    /* the image can be allocated by any means and av_image_alloc() is
    * just the most convenient way if av_malloc() is to be used */
    ret = av_image_alloc(frame->data, frame->linesize, c->width, c->height, c->pix_fmt, 32);
    if (ret < 0)
    {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        exit(1);
    }


    int video_size = 0;
    *out_data = (unsigned char*)malloc(MAX_VIDEO_BUF_SIZE);
    
    int buf_indx = 0;
    int flags = 0;flags |= 0;
    tjhandle handle = tjInitDecompress();
    for (i = 0; i < in_num; i++)
    {
        //jpeg图片解压到frame中
        
        ret = tjDecompressToYUV2(handle, in_datas + buf_indx, sizes[i], &(frame->data[0][0]), 1920, 32, 1080, flags);
        
        if (ret < 0)
        {
            printf("compress to jpeg failed: %s\n", tjGetErrorStr());
        }
    
        
        buf_indx = buf_indx + sizes[i];
        
        av_init_packet(&pkt);
        
        pkt.data = NULL;    // packet data will be allocated by the encoder
        pkt.size = 0;

        frame->pts = i;
        
        //将frame压缩到h265视频
        
        ret = avcodec_encode_video2(c, &pkt, frame, &got_output);
        if (ret < 0)
        {
            fprintf(stderr, "Error encoding frame\n");
            return ret;
        }

        if (got_output)
        {
           //fwrite(pkt.data, 1, pkt.size, f);
           if(video_size + pkt.size > MAX_VIDEO_BUF_SIZE)
           {
               fprintf(stderr, "overflow video buf size\n");
               break;
           }
        
           memcpy(*out_data + video_size, pkt.data, pkt.size);
        
           video_size = video_size + pkt.size;
           av_packet_unref(&pkt);
        }
    }

    /* get the delayed frames */
    for (got_output = 1; got_output; i++) 
    {
        fflush(stdout);

        
        ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
        
        if (ret < 0) 
        {
            fprintf(stderr, "Error encoding frame\n");
            return ret;
        }

        if (got_output) 
        {
           if(video_size + pkt.size > MAX_VIDEO_BUF_SIZE)
           {
               fprintf(stderr, "overflow video buf size\n");
               break;
           }
        
           memcpy(*out_data + video_size, pkt.data, pkt.size);
           video_size = video_size + pkt.size;
           av_packet_unref(&pkt);
        }
    }

    *out_size = video_size;

        
    avcodec_close(c);
    av_free(c);
    av_frame_free(&frame);
}

