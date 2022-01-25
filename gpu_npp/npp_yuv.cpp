#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <fcntl.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

/*
ffmpeg 关键字 pix_fmt支持的格式代码在：
~/work/FFmpeg$ vim libavutil/pixdesc.c +374
*/

//ffmpeg -i test.jpg  -pix_fmt nv12 test.yuv
//ffmpeg -i test.jpg -s 1920x1080 -pix_fmt nv12 test.yuv

//g++ yuv.cpp -L /usr/local/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -o yuv

int main()
{
    int width  = 1920;
    int height = 1080;

    FILE *fp_nv12 = fopen("./test.yuv", "rb+");

    unsigned char *nv12_data_buffer = (unsigned char *) malloc(width * height * 3.0 / 2.0);

    fread(nv12_data_buffer, 1, width * height * 3.0 / 2.0, fp_nv12);
    
    
    //opencv test
    Mat picNV12 = Mat(height * 3.0 / 2.0, width, CV_8UC1, nv12_data_buffer);
    Mat picBGR;
    cvtColor(picNV12, picBGR, COLOR_YUV2BGRA_NV12);
    imwrite("test_opencv.bmp", picBGR);
    
    //nppi test
    
    //nv12空间申请
    int nv12_pitch;
    Npp8u* nv12 = nppiMalloc_8u_C1(width, height * 3.0 / 2.0, &nv12_pitch);
    
    cudaMemcpy2D(nv12,                  nv12_pitch,
                 nv12_data_buffer,      width,      width,  height * 3.0 / 2.0,
                 cudaMemcpyHostToDevice);
         
    Npp8u *pNV12Src[2] = {nv12,
                          nv12 + (nv12_pitch * height) };
                          
    //bgr空间申请
    int bgr_pitch;
    Npp8u* bgr = nppiMalloc_8u_C3(width, height, &bgr_pitch);
    
    //npp nv12 to bgr
    NppiSize size = { width, height };
    nppiNV12ToBGR_8u_P2C3R(pNV12Src, nv12_pitch,
                           bgr,      bgr_pitch,
                           size);
                           
    
    cudaMemcpy2D(picBGR.data, picBGR.step, bgr, bgr_pitch, width * 3, height, cudaMemcpyDeviceToHost);
    
    imwrite("test_nppi.bmp", picBGR);
    
    return 0;
}
