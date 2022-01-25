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
#define SRC_SIZE_W 1920
#define SRC_SIZE_H 1080
#define DST_SIZE_W 512
#define DST_SIZE_H 512
#define IMG_CHANNEL 3


int main()
{
    cudaError_t cuRet;

    /* 设置显卡,构建上下文 */
    cuRet = cudaSetDevice( 0 );
    assert( cuRet == cudaSuccess );

    Mat org_img_mat;
    org_img_mat = imread("1.jpg", CV_LOAD_IMAGE_COLOR);

    //测试一下内存的BGR图片使用方式
    unsigned char* psrc = (unsigned char*)malloc(SRC_SIZE_W*SRC_SIZE_H*IMG_CHANNEL); 
    memset(psrc, 0, SRC_SIZE_W*SRC_SIZE_H*IMG_CHANNEL);
    memcpy(psrc, org_img_mat.data, SRC_SIZE_W*SRC_SIZE_H*IMG_CHANNEL);
    Mat src_mat(1080, 1920, org_img_mat.type(), psrc);
    imwrite("mc.jpg",src_mat);

    int image_a_pitch;
    NppiSize image_a_size = {.width = SRC_SIZE_W, .height = SRC_SIZE_H};
    NppiRect image_a_roi = {.x = 0, .y = 0, .width = SRC_SIZE_W, .height = SRC_SIZE_H};
    Npp8u* image_a = nppiMalloc_8u_C3(SRC_SIZE_W, SRC_SIZE_H, &image_a_pitch);
    
    int image_b_pitch;
    NppiSize image_b_size = {.width = DST_SIZE_W, .height = DST_SIZE_H};
    NppiRect image_b_roi = {.x = 0, .y = 0, .width = DST_SIZE_W, .height = DST_SIZE_H};
    Npp8u* image_b = nppiMalloc_8u_C3(DST_SIZE_W, DST_SIZE_H, &image_b_pitch);
    

    /* 建目标图 */
    Mat dst_mat;
    dst_mat.create(DST_SIZE_H, DST_SIZE_W, org_img_mat.type());
    
    double time_escp = 0;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for (int i = 0; i < 100; i ++)
    {
        cuRet = cudaMemcpy2D(image_a, image_a_pitch, src_mat.data,  src_mat.step, src_mat.cols * src_mat.elemSize() , src_mat.rows , cudaMemcpyHostToDevice);
       assert( cuRet == cudaSuccess );
        //cudaMemcpy2D(image_a, image_a_pitch, psrc,  SRC_SIZE_W*IMG_CHANNEL, SRC_SIZE_W*IMG_CHANNEL , SRC_SIZE_H , cudaMemcpyHostToDevice);
        //执行resize
        NppStatus result = nppiResize_8u_C3R(image_a, image_a_pitch, image_a_size, image_a_roi, image_b, image_b_pitch, image_b_size, image_b_roi, NPPI_INTER_LANCZOS);
        if (result != NPP_SUCCESS)
        {
            std::cerr << "Error executing Resize -- code: " << result << std::endl;
        }

        cuRet = cudaMemcpy2D(dst_mat.data, dst_mat.step, image_b, image_b_pitch, image_b_size.width * IMG_CHANNEL, image_b_size.height, cudaMemcpyDeviceToHost);
        assert( cuRet == cudaSuccess );
    }
    //cudaStreamSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    time_escp += (t_end.tv_sec  - t_start.tv_sec);
    time_escp += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;
    
    printf("use time:%f \r\n",time_escp);
    
    imwrite("./2.jpg",dst_mat);
    nppiFree( image_a );
    nppiFree( image_b );

    cudaDeviceReset();

    return 0;
}
