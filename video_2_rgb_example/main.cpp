//https://blog.csdn.net/qq_28840229/article/details/84572231
#include <iostream>
using namespace std;
 
extern "C"{
    #include "libavcodec/avcodec.h"
    #include "libavformat/avformat.h"
    #include "libavutil/pixfmt.h"
    #include "libswscale/swscale.h"
}
#include <stdio.h>
void SaveFrame(AVFrame* pFrame,int width,int height,int index)
{
    FILE* pFile;
    char szFilename[32];
    int y;
    sprintf(szFilename,"frame%d.rgb",index);
    pFile=fopen(szFilename,"wb");
 
    if(pFile==NULL)
    {
        return;
    }
    fprintf(pFile,"P6%d %d 255",width,height);
 
    for(y=0;y<height;y++)
    {
        fwrite(pFrame->data[0]+y*pFrame->linesize[0],1,width*3,pFile);
    }
    fclose(pFile);
}
int main()
{
    char *file_path = "test.hevc";
    AVFormatContext *pFormatCtx;  //FFMPEG所有的操作都要通过这个AVFormatContext
    AVCodecContext *pCodecCtx;   //AVCodecContext是一个描述编解码器上下文的数据结构，包含了众多编解码器需要的参数信息
    AVCodec *pCodec;   //存储编解码器信息的结构体
    AVFrame *pFrame, *pFrameRGB;//一般用于存储原始数据（即非压缩数据，例如对视频来说是YUV，RGB，对音频来说是PCM）
    AVPacket *packet;   //存储压缩编码数据相关信息的结构体
    uint8_t *out_buffer;
 
    static struct SwsContext *img_convert_ctx;  //主要用于视频图像的转换
 
    int videoStream, i, numBytes;
    int ret, got_picture;
 
    av_register_all(); //初始化FFMPEG  调用了这个才能正常适用编码器和解码器
 
    //Allocate an AVFormatContext.
    pFormatCtx = avformat_alloc_context();//进行初始化
 
 
    if (avformat_open_input(&pFormatCtx, file_path, NULL, NULL) != 0) { //媒体打开函数解析
    //能够获取到足够多的关于文件、流和CODEC的信息，并将这些信息填充到pFormatCtx
        printf("can't open the file. ");
        return -1;
     }
 
     if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {//该函数可以读取一部分视音频数据并且获得一些相关的信息
         printf("Could't find stream infomation. ");
         return -1;
      }
 
      videoStream = -1;
 
      //循环查找视频中包含的流信息，直到找到视频类型的流
      //便将其记录下来 保存到videoStream变量中
      //这里我们现在只处理视频流  音频流先不管他
      for (i = 0; i < pFormatCtx->nb_streams; i++) {
           if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStream = i;
           }
      }
 
       //如果videoStream为-1 说明没有找到视频流
       if (videoStream == -1) {
             printf("Didn't find a video stream.");
             return -1;
       }
 
        //查找解码器
       pCodecCtx = pFormatCtx->streams[videoStream]->codec;
       pCodec = avcodec_find_decoder(pCodecCtx->codec_id); //用于查找FFmpeg的解码器
 
       if (pCodec == NULL) {
            printf("Codec not found. ");
            return -1;
       }
 
 
        //打开解码器
        if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {  //用于初始化一个视音频编解码器
             printf("Could not open codec.");
             return -1;
        }
 
        pFrame = av_frame_alloc();
        pFrameRGB = av_frame_alloc();
        img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height,
                             pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height,
                             AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL); //1.图像色彩空间转换；2.分辨率缩放；3.前后图像滤波处理。
 
        numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pCodecCtx->width,pCodecCtx->height);
 
        out_buffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));//内存分配
        avpicture_fill((AVPicture *) pFrameRGB, out_buffer, AV_PIX_FMT_BGR24,
                          pCodecCtx->width, pCodecCtx->height); //为已经分配的空间的结构体AVPicture挂上一段用于保存数据的空间，这个结构体中有一个指针数组data[4]，挂在这个数组里
 
        int y_size = pCodecCtx->width * pCodecCtx->height;
 
        packet = (AVPacket *) malloc(sizeof(AVPacket)); //分配一个packet
        av_new_packet(packet, y_size); //分配packet的数据
 
        av_dump_format(pFormatCtx, 0, file_path, 0); //输出视频信息
 
        int index = 0;
        while (1)
        {
           if (av_read_frame(pFormatCtx, packet) < 0) //读取码流中的音频若干帧或者视频一帧
           {
                break; //这里认为视频读取完了
           }
 
           if (packet->stream_index == videoStream) {
               ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture,packet);//从AVPacket中解码数据到AVFrame中
 
               if (ret < 0) {
                       printf("decode error.");
                       return -1;
               }
 
                if (got_picture) {
                    sws_scale(img_convert_ctx,(uint8_t const * const *) pFrame->data,
                                 pFrame->linesize, 0, pCodecCtx->height, pFrameRGB->data,
                                             pFrameRGB->linesize);
 
                    SaveFrame(pFrameRGB, pCodecCtx->width,pCodecCtx->height,index++); //保存图片
                    if (index > 50) return 0; //这里我们就保存50张图片
                 }
             }
             av_free_packet(packet);
         }
         av_free(out_buffer);
         av_free(pFrameRGB);
         avcodec_close(pCodecCtx);
         avformat_close_input(&pFormatCtx);
 
         return 0;
}

