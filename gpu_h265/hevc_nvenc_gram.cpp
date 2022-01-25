#include <stdio.h>
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
};


static void video_encode_example(const char *filename)
{

    AVCodec *codec;
    AVCodecContext *c = NULL;
    int i, ret, x, y, got_output;
    AVFrame *frame, *cpu_frame;
    AVPacket pkt;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    av_log_set_level(64);

    codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    c->bit_rate = 400000;
    c->width = 1920;
    c->height = 1080;
    c->time_base.num = 1;
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_CUDA;
    c->max_b_frames = 0;

    AVBufferRef *device_ref = NULL;
    AVBufferRef *hw_frames_ctx = NULL;
    AVBufferRef *hw_device_ref = NULL;
    ret = av_hwdevice_ctx_create(&hw_device_ref, AV_HWDEVICE_TYPE_CUDA,"CUDA", NULL, 0);
    if (ret < 0) {
         fprintf(stderr, "Cannot open the hardware device\n");
    }

     AVHWFramesContext  *frames_ctx;
     c->hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ref);
     if (!c->hw_frames_ctx)
         return;
     frames_ctx   = (AVHWFramesContext*)c->hw_frames_ctx->data;

     frames_ctx->format            = AV_PIX_FMT_CUDA;
     frames_ctx->sw_format         = AV_PIX_FMT_YUV420P;
     frames_ctx->width             = c->width;
     frames_ctx->height            = c->height;
     frames_ctx->initial_pool_size = 32;

     ret = av_hwframe_ctx_init(c->hw_frames_ctx);
     if (ret < 0)
         return;

    AVDictionary *param = 0;
    //H.264
    if (codec->id == AV_CODEC_ID_H264) {
        av_dict_set(&param, "preset", "medium", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
    }
    //H.265
    if (codec->id == AV_CODEC_ID_H265 || codec->id == AV_CODEC_ID_HEVC){
        //av_dict_set(&param, "x265-params", "qp=20", 0);
        av_dict_set(&param, "x265-params", "crf=25", 0);
        av_dict_set(&param, "preset", "fast", 0);
        av_dict_set(&param, "tune", "zero-latency", 0);
    }

    /* open it */

    if (avcodec_open2(c, NULL, &param) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        system("pause");
        exit(1);
    }



    FILE *f;
    f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    frame = av_frame_alloc();

    if (!frame)
    {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    printf("%d\r\n",__LINE__);
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;
    avcodec_default_get_buffer2(c, frame, 0);

    cpu_frame = av_frame_alloc();

    if (!cpu_frame)
    {
        fprintf(stderr, "Could not allocate video cpu_frame\n");
        exit(1);
    }
    cpu_frame->format = AV_PIX_FMT_YUV420P;
    cpu_frame->width  =  c->width;
    cpu_frame->height = c->height;


    ret = av_image_alloc(cpu_frame->data, cpu_frame->linesize, c->width, c->height, AV_PIX_FMT_YUV420P, 32);
    if (ret < 0) 
    {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        exit(1);
    }

    /* encode 1 second of video */
    for (i = 0; i < 500; i++) 
    {
        av_init_packet(&pkt);
        pkt.data = NULL;    // packet data will be allocated by the encoder
        pkt.size = 0;

        for (y = 0; y < c->height; y++) {
            for (x = 0; x < c->width; x++) {
                cpu_frame->data[0][y * cpu_frame->linesize[0] + x] = x + y + i * 3;
            }
        }
        for (y = 0; y < c->height / 2; y++) {
            for (x = 0; x < c->width / 2; x++) {
                cpu_frame->data[1][y * cpu_frame->linesize[1] + x] = 128 + y + i * 2;
                cpu_frame->data[2][y * cpu_frame->linesize[2] + x] = 64 + x + i * 5;
            }
        }

        if ((ret = av_hwframe_transfer_data(frame, cpu_frame, 0)) < 0)
        {
            fprintf(stderr, "Error transferring the data to system memory\n");
            exit(1);
        }
        //也可以直接用cudaMemset设置内存图片不需要用av_hwframe_transfer_data了
        //或者cudaMemcpy2D函数也可以从cpu内存拷贝数据到GPU里面
        //cudaMemset(&(frame->data[1][y * frame->linesize[1] + x]),128 + y + i * 2,1);
       // cudaMemset(&(frame->data[2][y * frame->linesize[2] + x]),64 + x + i * 5,1);

        frame->pts = i;
        /* encode the image */

        ret = avcodec_encode_video2(c, &pkt, frame, &got_output);
        if (ret < 0) {
            fprintf(stderr, "Error encoding frame\n");
            exit(1);
        }

        if (got_output) {
            printf("Write frame %3d (size=%5d)\n", i, pkt.size);
            fwrite(pkt.data, 1, pkt.size, f);
            av_packet_unref(&pkt);
        }

    }

    /* get the delayed frames */
    for (got_output = 1; got_output; i++) {
        fflush(stdout);

        ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
        if (ret < 0) {
            fprintf(stderr, "Error encoding frame\n");
            exit(1);
        }

        if (got_output) {
            printf("Write frame %3d (size=%5d)\n", i, pkt.size);
            fwrite(pkt.data, 1, pkt.size, f);
            av_packet_unref(&pkt);
        }
    }

    /* add sequence end code to have a real MPEG file */
    fwrite(endcode, 1, sizeof(endcode), f);
    fclose(f);

    avcodec_close(c);
    av_free(c);
    av_freep(&frame->data[0]);
    av_frame_free(&frame);
    printf("\n");
}

int main(int argc, char **argv)
{
    /* register all the codecs */
    avcodec_register_all();

    avcodec_register_all();

    avdevice_register_all();

    avfilter_register_all();
    av_register_all();
    avformat_network_init();

    video_encode_example("test_gram.hevc");

    system("pause");

    return 0;
}
