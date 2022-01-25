#include <stdio.h>
#include <time.h>
#include "wrapper.h"

#define MAX_VIDEO_BUF_SIZE  32* 1024*1024

void compress_2_h265(unsigned char* in_datas, int* sizes, int in_num, int width, int height, unsigned char* out_data, int *out_size)
{
    avcodec_register_all();
    avcodec_register_all();
    avdevice_register_all();
    avfilter_register_all();
    av_register_all();
    avformat_network_init();

    AVCodec *codec;
    AVCodecContext *c = NULL;
    int ret, i, got_output;
    AVFrame *frame;
    AVPacket pkt;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    //av_log_set_level(64);
    av_log_set_level(0);

    codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec)
    {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c)
    {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    c->bit_rate      = 400000;
    c->width         = width;
    c->height        = height;
    c->time_base.num = 1;
    c->gop_size      = 10;
    c->max_b_frames  = 1;
    c->pix_fmt       = AV_PIX_FMT_CUDA;
    c->max_b_frames  = 0;

    AVBufferRef *device_ref    = NULL;
    AVBufferRef *hw_frames_ctx = NULL;
    AVBufferRef *hw_device_ref = NULL;
    ret = av_hwdevice_ctx_create(&hw_device_ref, AV_HWDEVICE_TYPE_CUDA,"CUDA", NULL, 0);
    if (ret < 0)
    {
         fprintf(stderr, "Cannot open the hardware device\n");
    }

    AVHWDeviceContext*   hw_dev_context = (AVHWDeviceContext*)(hw_device_ref->data);
    AVCUDADeviceContext* cuda_dev_ctx   = (AVCUDADeviceContext*)(hw_dev_context->hwctx);
    AVHWFramesContext*   frames_ctx;
    c->hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ref);
    if (!c->hw_frames_ctx)
    {
        fprintf(stderr, "av_hwframe_ctx_init fail\n");
        system("pause");
        exit(1);
    }
    frames_ctx                    = (AVHWFramesContext*)c->hw_frames_ctx->data;
    frames_ctx->format            = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format         = AV_PIX_FMT_YUV420P;
    frames_ctx->width             = c->width;
    frames_ctx->height            = c->height;
    frames_ctx->initial_pool_size = 32;

    ret = av_hwframe_ctx_init(c->hw_frames_ctx);
    if (ret < 0)
    {
       fprintf(stderr, "av_hwframe_ctx_init fail\n");
       system("pause");
       exit(1);
    }
    AVDictionary *param = 0;
    //H.265
    if (codec->id == AV_CODEC_ID_H265 || codec->id == AV_CODEC_ID_HEVC){
        //av_dict_set(&param, "x265-params", "qp=20", 0);
        av_dict_set(&param, "x265-params", "crf=25", 0);
        av_dict_set(&param, "preset", "fast", 0);
        av_dict_set(&param, "tune", "zero-latency", 0);
    }

    if (avcodec_open2(c, NULL, &param) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        system("pause");
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame)
    {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;
    avcodec_default_get_buffer2(c, frame, 0);//从cuda中分配frame

    //ffmpeg申请好的gpu内存放到nvjpeg接收jpeg转yuv420的结果,ffmpeg拿这个结果继续做h265压缩
    //NvJpeg jpeg_dec(width, height, 0, frame, NVJPEG_OUTPUT_YUV);
    void *p_jpeg_dec = call_NvJpeg_Create(width, height, 0, frame, NVJPEG_OUTPUT_YUV);

    int video_size = 0;
    out_data = (unsigned char*)malloc(MAX_VIDEO_BUF_SIZE);
    
    int buf_indx = 0;
    for (i = 0; i < in_num; i++)
    {
        //jpeg图片解压到frame中
        //jpeg_dec.decompress(in_datas + buf_indx, sizes[i]);
        call_NvJpeg_Decompress(p_jpeg_dec, in_datas + buf_indx, sizes[i]);
        //print_hex_str(in_datas + buf_indx ,64);
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
            exit(1);
        }

        if (got_output)
        {
           //fwrite(pkt.data, 1, pkt.size, f);
           if(video_size + pkt.size > MAX_VIDEO_BUF_SIZE)
           {
               fprintf(stderr, "overflow video buf size\n");
               break;
           }
           memcpy(out_data + video_size, pkt.data, pkt.size);
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
            exit(1);
        }

        if (got_output) 
        {
           if(video_size + pkt.size > MAX_VIDEO_BUF_SIZE)
           {
               fprintf(stderr, "overflow video buf size\n");
               break;
           }
           memcpy(out_data + video_size, pkt.data, pkt.size);
           video_size = video_size + pkt.size;
           av_packet_unref(&pkt);
        }
    }

    *out_size = video_size;

    avcodec_close(c);
    av_free(c);
    av_frame_free(&frame);
    call_NvJpeg_Destroy(p_jpeg_dec);
}
