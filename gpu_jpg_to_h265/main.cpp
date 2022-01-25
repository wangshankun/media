#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <time.h>
#include "nvJPEG.hpp"
#include "nvJPEG_helper.hpp"

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

typedef  std::vector<char>  FileData;


static void video_encode_example(const char *filename)
{
    AVCodec *codec;
    AVCodecContext *c = NULL;
    int i, ret, x, y, got_output;
    AVFrame *frame, *cpu_frame;
    AVPacket pkt;
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    //av_log_set_level(64);
    av_log_set_level(0);

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

    AVHWDeviceContext*   hw_dev_context = (AVHWDeviceContext*)(hw_device_ref->data);
    AVCUDADeviceContext* cuda_dev_ctx   = (AVCUDADeviceContext*)(hw_dev_context->hwctx);

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
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;
    avcodec_default_get_buffer2(c, frame, 0);

    //ffmpeg申请好的gpu内存放到nvjpeg接收jpeg转yuv420的结果,ffmpeg拿这个结果继续做h265压缩
    NvJpeg jpeg_dec(1920, 1080, 0, &(cuda_dev_ctx->cuda_ctx), frame, NVJPEG_OUTPUT_YUV);
    
    
    std::vector<std::string> inputFiles;
    if (readInput("/home/shankun.shankunwan/si112/", inputFiles))
    {
        std::cerr << "Cannot open path dir: ";
    }

    printf("inputFiles.size():%d \r\n",inputFiles.size());
    for (unsigned int i = 0; i < inputFiles.size(); i++)
    {
        std::string &sFileName = inputFiles[i];
        FileData raw_data;
        std::ifstream input(sFileName, std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
            std::cerr << "Cannot open image: ";
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data.size() < file_size)
        {
            raw_data.resize(file_size);
        }

        if (!input.read(raw_data.data(), file_size))
        {
            std::cerr << "Cannot read from file: ";
        }
    
        jpeg_dec.decompress_to_gpubuf((unsigned char *)(&raw_data[0]), file_size);

        av_init_packet(&pkt);
        pkt.data = NULL;    // packet data will be allocated by the encoder
        pkt.size = 0;

        frame->pts = i;
        
        ret = avcodec_encode_video2(c, &pkt, frame, &got_output);
        if (ret < 0) {
            fprintf(stderr, "Error encoding frame\n");
            exit(1);
        }

        if (got_output) {
 //           printf("Write frame %3d (size=%5d)\n", i, pkt.size);
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
//            printf("Write frame %3d (size=%5d)\n", i, pkt.size);
            fwrite(pkt.data, 1, pkt.size, f);
            av_packet_unref(&pkt);
        }
    }

    fwrite(endcode, 1, sizeof(endcode), f);
    fclose(f);

    avcodec_close(c);
    av_free(c);
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

    double elapsed = 0;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    video_encode_example("test.hevc");

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("elapsed time:%f\r\n",elapsed);

    return 0;
}
