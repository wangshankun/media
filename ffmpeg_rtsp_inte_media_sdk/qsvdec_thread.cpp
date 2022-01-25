#include <opencv2/opencv.hpp>
#include <chrono> 
#include <iostream> 
#include <thread>
#include <time.h>    
extern "C"
{
#include "config.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavcodec/avcodec.h"
#include "libavutil/buffer.h"
#include "libavutil/error.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_qsv.h"
#include "libavutil/mem.h"
#include "libavdevice/avdevice.h"
#include "libavfilter/avfilter.h"
#include "libavutil/avutil.h"
#include "libavutil/time.h"
#include "libswscale/swscale.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"
#include "libavutil/common.h"
}

using namespace std;
using namespace cv;

typedef struct DecodeContext {
    AVBufferRef *hw_device_ref;
} DecodeContext;


int out_img_w  = 448;
int out_img_h  = 448;


static int get_format(AVCodecContext *avctx, const enum AVPixelFormat *pix_fmts)
{
    while (*pix_fmts != AV_PIX_FMT_NONE) {
        if (*pix_fmts == AV_PIX_FMT_QSV) {
            DecodeContext *decode = avctx->opaque;
            AVHWFramesContext  *frames_ctx;
            AVQSVFramesContext *frames_hwctx;
            int ret;

            /* create a pool of surfaces to be used by the decoder */
            avctx->hw_frames_ctx = av_hwframe_ctx_alloc(decode->hw_device_ref);
            if (!avctx->hw_frames_ctx)
                return AV_PIX_FMT_NONE;
            frames_ctx   = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
            frames_hwctx = frames_ctx->hwctx;

            frames_ctx->format            = AV_PIX_FMT_QSV;
            frames_ctx->sw_format         = avctx->sw_pix_fmt;
            frames_ctx->width             = FFALIGN(avctx->coded_width,  32);
            frames_ctx->height            = FFALIGN(avctx->coded_height, 32);
            frames_ctx->initial_pool_size = 32;

            frames_hwctx->frame_type = MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET;

            ret = av_hwframe_ctx_init(avctx->hw_frames_ctx);
            if (ret < 0)
                return AV_PIX_FMT_NONE;

            return AV_PIX_FMT_QSV;
        }

        pix_fmts++;
    }

    fprintf(stderr, "The QSV pixel format not offered in get_format()\n");

    return AV_PIX_FMT_NONE;
}

AVFrame *AllocFrame(enum AVPixelFormat format, int width, int height)
{
    AVFrame *frame = av_frame_alloc();
    if (!frame) return NULL;
    frame->format = format;
    frame->width = width;
    frame->height = height;
    frame->pts = 0;

    int ret = av_image_alloc(frame->data, frame->linesize, frame->width, frame->height, (enum AVPixelFormat)frame->format, 1);

    if (ret < 0)
    {
        av_frame_free(&frame);
        return NULL;
    }
    return frame;
}


void cap_stream_task(int thread_id, const string& rtsp)
{
    AVFormatContext *input_ctx = NULL;
    AVStream *video_st = NULL;
    AVCodecContext *decoder_ctx = NULL;
    const AVCodec *decoder;
    AVPacket pkt = { 0 };

    DecodeContext decode = { NULL };

    int ret, i, videoindex = -1;

    av_register_all();
    avformat_network_init(); 
    AVDictionary *optionsDict = NULL;
    av_dict_set(&optionsDict, "rtsp_transport", "tcp", 0);
    av_dict_set(&optionsDict, "stimeout", "2000000", 0);

    ret = avformat_open_input(&input_ctx, rtsp.c_str(), 0, &optionsDict);
    if (ret < 0)
    {
        printf("Cannot open input file! \r\n");
        goto finish;
    }
    if ((ret = avformat_find_stream_info(input_ctx, NULL)) < 0) 
    {
        printf("Could not open find stream info \n");
        goto finish;
    }
    for (i = 0; i < input_ctx->nb_streams; i++)
    {
        av_dump_format(input_ctx, i, rtsp.c_str(), 0);
    }
    for (i = 0; i < input_ctx->nb_streams; i++)
    {
        AVStream *st = input_ctx->streams[i];
        if ((input_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) && (videoindex < 0))
        {
            videoindex = i;
            video_st   = st;
        }
    }

    ret = av_hwdevice_ctx_create(&decode.hw_device_ref, AV_HWDEVICE_TYPE_QSV,
                                 "auto", NULL, 0);
    decoder = avcodec_find_decoder_by_name("h264_qsv");
    decoder_ctx = avcodec_alloc_context3(decoder);
    decoder_ctx->codec_id = AV_CODEC_ID_H264;
    if (video_st->codecpar->extradata_size) {
        decoder_ctx->extradata = av_mallocz(video_st->codecpar->extradata_size +
                                            AV_INPUT_BUFFER_PADDING_SIZE);
        if (!decoder_ctx->extradata) {
            ret = AVERROR(ENOMEM);
            goto finish;
        }
        memcpy(decoder_ctx->extradata, video_st->codecpar->extradata,
               video_st->codecpar->extradata_size);
        decoder_ctx->extradata_size = video_st->codecpar->extradata_size;
    }
    decoder_ctx->refcounted_frames = 1;
    decoder_ctx->opaque      = &decode;
    decoder_ctx->get_format  = get_format;
    ret = avcodec_open2(decoder_ctx, NULL, NULL);

 
    AVFrame * frame    = av_frame_alloc();
    AVFrame * sw_frame = av_frame_alloc();
    if (!frame || !sw_frame) {printf("malloc frame fail\r\n");}

    int video_st_w = video_st->codec->width;
    int video_st_h = video_st->codec->height;
    

    SwsContext *img_convert_ctx = sws_getContext(video_st_w, \
                                                 video_st_h, \
                                                 AV_PIX_FMT_NV12, \
                                                 out_img_w, \
                                                 out_img_h, \
                                                 AV_PIX_FMT_BGR24, \
                                                 SWS_BICUBIC, NULL, NULL, NULL);
    AVFrame *bgr_frame = AllocFrame(AV_PIX_FMT_BGR24, \
                                      out_img_w, \
                                      out_img_h);

    int        bgr_size   = av_image_get_buffer_size(bgr_frame->format, bgr_frame->width,
                                    bgr_frame->height, 1);
    uint8_t *  bgr_buffer = av_malloc(bgr_size);
    
//    double elapsed;
//    struct timespec start, finish;
    long count = 0;
    while (1)
    {
        //clock_gettime(CLOCK_MONOTONIC, &start);
        ret = av_read_frame(input_ctx, &pkt);
        if (pkt.stream_index == video_st->index)
        {
            ret = avcodec_send_packet(decoder_ctx, &pkt);
            while (ret >= 0) 
            {
                ret = avcodec_receive_frame(decoder_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;

                if(count%2 == 0 )//跳帧
                {
                    /* AV_PIX_FMT_QSV 到 AV_PIX_FMT_NV12*/ 
                    ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                    /*从AV_PIX_FMT_NV12 到 AV_PIX_FMT_BGR24*/
                    sws_scale(img_convert_ctx, (const unsigned char* const*)sw_frame->data, sw_frame->linesize, 0, sw_frame->height, bgr_frame->data, bgr_frame->linesize);
                    ret = av_image_copy_to_buffer(bgr_buffer, bgr_size,
                                                  (const uint8_t * const *)bgr_frame->data,
                                                  (const int *)bgr_frame->linesize, bgr_frame->format,
                                                  bgr_frame->width, bgr_frame->height, 1);

                    Mat my_mat(bgr_frame->width, bgr_frame->height, CV_8UC3, bgr_buffer);
                    //char s[64];
                    //sprintf(s, "Display window %d", thread_id);
                    //imshow( s, my_mat);
                    //waitKey(0);
                    if(count%1000 == 0)//每N帧保存一张图,加时间戳与视频流时间对比，看解码性能是否满足
                    {
                        time_t t = time(NULL);
                        struct tm tm = *localtime(&t);
                        char s[64] = "";
                        sprintf(s, "%d_%d_%d_%d_%d.bmp", thread_id,count,tm.tm_hour, tm.tm_min, tm.tm_sec);
                        imwrite(s, my_mat);
                    }
                    av_frame_unref(sw_frame);
                    av_frame_unref(frame);
                }
                count++;
            }
        }

        av_packet_unref(&pkt);

    //    clock_gettime(CLOCK_MONOTONIC, &finish);
    //    elapsed = (finish.tv_sec - start.tv_sec);
    //    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    //    printf("elapsed time:%f\r\n",elapsed);
    }
finish:
    if (ret < 0) 
    {
        char buf[1024];
        av_strerror(ret, buf, sizeof(buf));
        printf("%s\n", buf);
    }

    avformat_close_input(&input_ctx);
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_frame_free(&bgr_frame);
    av_freep(&bgr_buffer);
    avcodec_free_context(&decoder_ctx);
    av_buffer_unref(&decode.hw_device_ref);
}

#define thread_num 8
string rtsp = "rtsp://admin:12345678@30.1.53.16:554";
int main(int argc, const char *argv[])
{
    thread threads[thread_num];

    for (int i = 0; i < thread_num; i++) 
    {
        threads[i] = thread(cap_stream_task, i, ref(rtsp));
    }

    for (auto& t: threads) {
        t.join();
    }


    return EXIT_SUCCESS;
}
