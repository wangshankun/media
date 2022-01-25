#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <fcntl.h>
#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>
#include <FreeImage.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

/*
    NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;
    if (nScaleFactor >= 1.f) eInterploationMode = NPPI_INTER_LANCZOS;
    NPP_CHECK_NPP(nppiResize_8u_C1R(..., eInterploationMode));
*/

#define SRC_SIZE_W 1920
#define SRC_SIZE_H 1080
#define DST_SIZE_W 512
#define DST_SIZE_H 512
#define IMG_CHANNEL 3
#define BATCH_SIZE  32

int main()
{
    cudaError_t cuRet;

    NppiResizeBatchCXR *p_npp_batch_r = (NppiResizeBatchCXR *)malloc(sizeof(NppiResizeBatchCXR));
    
    /* 设置显卡,构建上下文 */
    cuRet = cudaSetDevice( 0 );
    assert( cuRet == cudaSuccess );


    unsigned char* host_imgs = (unsigned char*)malloc(SRC_SIZE_W * SRC_SIZE_H * IMG_CHANNEL * BATCH_SIZE);

    for (int i = 1; i < 33; i++)
    {
        char img_name [50] = {0};
        sprintf(img_name, "/home/shankun.shankunwan/work/dali/images/sc/sc_%d.jpg", i);
        Mat src;
        src = imread(img_name, CV_LOAD_IMAGE_COLOR);
        memcpy(host_imgs + (i - 1) * SRC_SIZE_W * SRC_SIZE_H * IMG_CHANNEL, src.data, SRC_SIZE_W * SRC_SIZE_H * IMG_CHANNEL);
    }
    
    unsigned char *device_src_data = NULL;
    unsigned char *device_dst_data = NULL;
    cuRet = cudaMalloc((void**)&device_src_data, SRC_SIZE_W * SRC_SIZE_H * IMG_CHANNEL * BATCH_SIZE);
    assert( cuRet == cudaSuccess );
    cuRet = cudaMalloc((void**)&device_dst_data, DST_SIZE_W * DST_SIZE_H * IMG_CHANNEL * BATCH_SIZE);
    assert( cuRet == cudaSuccess );

    cuRet = cudaMemcpy(device_src_data, host_imgs, SRC_SIZE_W * SRC_SIZE_H * IMG_CHANNEL * BATCH_SIZE, cudaMemcpyHostToDevice);
    assert( cuRet == cudaSuccess );
    
    p_npp_batch_r -> pSrc     = device_src_data;
    p_npp_batch_r -> nSrcStep = SRC_SIZE_W * SRC_SIZE_H * IMG_CHANNEL;
    p_npp_batch_r -> pDst     = device_dst_data;
    p_npp_batch_r -> nDstStep = DST_SIZE_W * DST_SIZE_H * IMG_CHANNEL;
    
    NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;

    NppiSize image_a_size = {.width = SRC_SIZE_W, .height = SRC_SIZE_H};
    NppiRect image_a_roi = {.x = 0, .y = 0, .width = SRC_SIZE_W, .height = SRC_SIZE_H};

    NppiSize image_b_size = {.width = DST_SIZE_W, .height = DST_SIZE_H};
    NppiRect image_b_roi = {.x = 0, .y = 0, .width = DST_SIZE_W, .height = DST_SIZE_H};

    //执行resize
    NppStatus result = nppiResizeBatch_8u_C3R(image_a_size, image_a_roi, image_b_size, image_b_roi,  eInterploationMode, p_npp_batch_r, BATCH_SIZE);
    if (result != NPP_SUCCESS)
    {
        std::cerr << "Error executing Resize -- code: " << result << std::endl;
    }

    unsigned char* host_dst_imgs = (unsigned char*)malloc(DST_SIZE_W * DST_SIZE_H * IMG_CHANNEL * BATCH_SIZE);
    memset(host_dst_imgs, 0, DST_SIZE_W * DST_SIZE_H * IMG_CHANNEL * BATCH_SIZE);
    
    cuRet = cudaMemcpy(host_dst_imgs, device_dst_data, DST_SIZE_W * DST_SIZE_H * IMG_CHANNEL * BATCH_SIZE, cudaMemcpyDeviceToHost);
    printf("%d\r\n",cuRet);
    assert( cuRet == cudaSuccess );
    
    for (int i = 1; i < 33; i++)
    {
        Mat mat(DST_SIZE_H, DST_SIZE_W, CV_8UC3);
        
        memcpy(mat.data, host_dst_imgs + (i - 1) * DST_SIZE_W * DST_SIZE_W * IMG_CHANNEL, DST_SIZE_W * DST_SIZE_W * IMG_CHANNEL);
        
        char img_name [50] = {0};
        sprintf(img_name, "/home/shankun.shankunwan/work/dali/images/sc/ot_%d.bmp", i);
        imwrite(img_name, mat);
    }
    
    return 0;
}
