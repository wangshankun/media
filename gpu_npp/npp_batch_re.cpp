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

#define SRC_SIZE_W 1920
#define SRC_SIZE_H 1080
#define DST_SIZE_W 512
#define DST_SIZE_H 512
#define IMG_CHANNEL 3
#define BATCH_SIZE  32

int main()
{
    cudaError_t cuRet;

    /* 设置显卡,构建上下文 */
    cuRet = cudaSetDevice( 0 );
    assert( cuRet == cudaSuccess );

    unsigned char *device_src_data = NULL;
    unsigned char *device_dst_data = NULL;

    int                          image_src_pitch[BATCH_SIZE];
    NppiSize                     image_src_size_array[BATCH_SIZE];
    NppiRect                     image_src_roi_array[BATCH_SIZE];
    Npp8u*                       image_src_array[BATCH_SIZE];
    int                          image_dst_pitch[BATCH_SIZE];
    NppiSize                     image_dst_size_array[BATCH_SIZE];
    NppiRect                     image_dst_roi_array[BATCH_SIZE];
    Npp8u*                       image_dst_array[BATCH_SIZE];
    NppiImageDescriptor          batch_image_src_descriptor[BATCH_SIZE];
    NppiImageDescriptor          batch_image_dst_descriptor[BATCH_SIZE];
    NppiResizeBatchROI_Advanced  batch_roi_descriptor[BATCH_SIZE];
    
    NppiImageDescriptor*         dev_batch_image_src_descriptor =  NULL;
    NppiImageDescriptor*         dev_batch_image_dst_descriptor =  NULL;
    NppiResizeBatchROI_Advanced* dev_batch_roi_descriptor       =  NULL;
    
    cudaMalloc((void**)&dev_batch_image_src_descriptor, BATCH_SIZE * sizeof(NppiImageDescriptor));
    cudaMalloc((void**)&dev_batch_image_dst_descriptor, BATCH_SIZE * sizeof(NppiImageDescriptor));
    cudaMalloc((void**)&dev_batch_roi_descriptor      , BATCH_SIZE * sizeof(NppiResizeBatchROI_Advanced));
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        char img_name [128] = {0};
        sprintf(img_name, "/home/shankun.shankunwan/work/dali/images/sc/sc_%d.jpg", i);
        Mat src_mat;
        src_mat = imread(img_name, CV_LOAD_IMAGE_COLOR);
        
        image_src_size_array[i]  = {.width = SRC_SIZE_W, .height = SRC_SIZE_H};
        image_src_roi_array[i]   = {.x = 0, .y = 0, .width = SRC_SIZE_W, .height = SRC_SIZE_H};
        image_src_array[i]       = nppiMalloc_8u_C3(SRC_SIZE_W, SRC_SIZE_H, &image_src_pitch[i]);

        cudaMemcpy2DAsync(image_src_array[i], image_src_pitch[i], src_mat.data,  src_mat.step, 
            src_mat.cols * src_mat.elemSize() , src_mat.rows , cudaMemcpyHostToDevice);

        image_dst_size_array[i]  = {.width = DST_SIZE_W, .height = DST_SIZE_H};
        image_dst_roi_array[i]   = {.x = 0, .y = 0, .width = DST_SIZE_W, .height = DST_SIZE_H};
        image_dst_array[i]       = nppiMalloc_8u_C3(DST_SIZE_W, DST_SIZE_H, &image_dst_pitch[i]);

        batch_image_src_descriptor[i].pData = image_src_array[i];
        batch_image_src_descriptor[i].nStep = image_src_pitch[i];
        batch_image_src_descriptor[i].oSize = image_src_size_array[i];
        batch_image_dst_descriptor[i].pData = image_dst_array[i];
        batch_image_dst_descriptor[i].nStep = image_dst_pitch[i];
        batch_image_dst_descriptor[i].oSize = image_dst_size_array[i];
        batch_roi_descriptor[i].oSrcRectROI = image_src_roi_array[i];
        batch_roi_descriptor[i].oDstRectROI = image_dst_roi_array[i];
    }
    
    cudaMemcpy(dev_batch_image_src_descriptor, batch_image_src_descriptor, BATCH_SIZE * sizeof(NppiImageDescriptor)        , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_batch_image_dst_descriptor, batch_image_dst_descriptor, BATCH_SIZE * sizeof(NppiImageDescriptor)        , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_batch_roi_descriptor      , batch_roi_descriptor      , BATCH_SIZE * sizeof(NppiResizeBatchROI_Advanced), cudaMemcpyHostToDevice);

    NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;
    //执行resize
    NppStatus result = nppiResizeBatch_8u_C3R_Advanced(SRC_SIZE_W, SRC_SIZE_H, dev_batch_image_src_descriptor, dev_batch_image_dst_descriptor, dev_batch_roi_descriptor, BATCH_SIZE, eInterploationMode);
    if (result != NPP_SUCCESS)
    {
        std::cerr << "Error executing Resize -- code: " << result << std::endl;
    }

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        Mat dst_mat(DST_SIZE_H, DST_SIZE_W, CV_8UC3);
        cudaMemcpy2D(dst_mat.data, dst_mat.step, image_dst_array[i], image_dst_pitch[i], image_dst_size_array[i].width * IMG_CHANNEL, image_dst_size_array[i].height, cudaMemcpyDeviceToHost);
        char img_name [128] = {0};
        sprintf(img_name, "/home/shankun.shankunwan/work/dali/images/sc/ot_%d.bmp", i);
        imwrite(img_name, dst_mat);
    }

    return 0;
}
