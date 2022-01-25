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
#include "opencv2/opencv.hpp"
#include <time.h>
#include <vector>


using namespace cv;
using namespace std;

#define MAX_SRC_SIZE_W 1920
#define MAX_SRC_SIZE_H 1500

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        if (status != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << status << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

typedef  struct resize_item {
    void* data;
    int   width;
    int   hight;
    int   step;
}mat_st;

enum ImgType {U8C3, U8C4};

class BatchGpuResize {
    public:
        //int Init(int max_batch, std::shared_ptr<BufPool> dev_bgra, std::shared_ptr<BufPool> dev_resize_bgra);
        int Init(int max_batch);
        int AddItem(void* src_data_ptr, int src_width, int src_hight, int src_step,
                    void* dst_data_ptr, int dst_width, int dst_hight, int dst_step, ImgType img_type);
        int DoResize(ImgType img_type,  NppiInterpolationMode resize_type);
        int GetElemSize(ImgType img_type);
        int Destroy();
    private:
        int                          batch_size;
        int                          max_batch_size;
        NppiImageDescriptor*         host_image_src_descriptors = NULL;
        NppiImageDescriptor*         host_image_dst_descriptors = NULL;
        NppiResizeBatchROI_Advanced* host_roi_descriptors       = NULL;
        NppiImageDescriptor*         dev_image_src_descriptors  = NULL;
        NppiImageDescriptor*         dev_image_dst_descriptors  = NULL;
        NppiResizeBatchROI_Advanced* dev_roi_descriptors        = NULL;
        mat_st*                      host_dst_mat               = NULL;
};

int BatchGpuResize::Destroy() {
    free(host_image_src_descriptors);
    free(host_image_dst_descriptors);
    free(host_roi_descriptors);

    CHECK(cudaFree(dev_image_src_descriptors));
    CHECK(cudaFree(dev_image_dst_descriptors));
    CHECK(cudaFree(dev_roi_descriptors));
    
    free(host_dst_mat);
    
    return 0;
}

int BatchGpuResize::GetElemSize(ImgType img_type) {
    int elem_size;
    if (img_type == U8C3) {
        elem_size = 3;
    }
    else if (img_type == U8C4) {
        elem_size = 4;
    }
    else {
        assert(false);
    }
    return elem_size;
}

/*
int BatchGpuResize::Init(int max_batch, std::shared_ptr<BufPool> dev_bgra, std::shared_ptr<BufPool> dev_resize_bgra) {
    max_batch_size = max_batch;
    batch_size     = 0;
    CHECK(cudaMalloc((void**)&dev_image_src_descriptors, max_batch_size * sizeof(NppiImageDescriptor)));
    CHECK(cudaMalloc((void**)&dev_image_dst_descriptors, max_batch_size * sizeof(NppiImageDescriptor)));
    CHECK(cudaMalloc((void**)&dev_roi_descriptors      , max_batch_size * sizeof(NppiResizeBatchROI_Advanced)));
    
    host_image_src_descriptors = (NppiImageDescriptor*)malloc(max_batch_size * sizeof(NppiImageDescriptor));
    host_image_dst_descriptors = (NppiImageDescriptor*)malloc(max_batch_size * sizeof(NppiImageDescriptor));
    host_roi_descriptors       = (NppiResizeBatchROI_Advanced*)malloc(max_batch_size * sizeof(NppiResizeBatchROI_Advanced));
    
    host_dst_mat               = (mat_st*)malloc(max_batch_size * sizeof(mat_st));

    for (int i = 0; i < max_batch_size; i++)
    {
        auto dev_src_img = dev_bgra->allocateOneBuf();
        auto dev_dst_img = dev_resize_bgra->allocateOneBuf();

        host_image_src_descriptors[i].pData = dev_src_img->getRawPointer();
        host_image_src_descriptors[i].nStep = dev_src_img->getPitch();
        host_image_dst_descriptors[i].pData = dev_dst_img->getRawPointer();
        host_image_dst_descriptors[i].nStep = dev_dst_img->getPitch();
    }

    return 0;
}
*/

int BatchGpuResize::Init(int max_batch) {
    max_batch_size = max_batch;
    batch_size     = 0;
    CHECK(cudaMalloc((void**)&dev_image_src_descriptors, max_batch_size * sizeof(NppiImageDescriptor)));
    CHECK(cudaMalloc((void**)&dev_image_dst_descriptors, max_batch_size * sizeof(NppiImageDescriptor)));
    CHECK(cudaMalloc((void**)&dev_roi_descriptors      , max_batch_size * sizeof(NppiResizeBatchROI_Advanced)));
    
    host_image_src_descriptors = (NppiImageDescriptor*)malloc(max_batch_size * sizeof(NppiImageDescriptor));
    host_image_dst_descriptors = (NppiImageDescriptor*)malloc(max_batch_size * sizeof(NppiImageDescriptor));
    host_roi_descriptors       = (NppiResizeBatchROI_Advanced*)malloc(max_batch_size * sizeof(NppiResizeBatchROI_Advanced));
    
    host_dst_mat               = (mat_st*)malloc(max_batch_size * sizeof(mat_st));

    for (int i = 0; i < max_batch_size; i++)
    {
        int pitch;
        host_image_src_descriptors[i].pData = nppiMalloc_8u_C3(MAX_SRC_SIZE_W, MAX_SRC_SIZE_H, &pitch);
        host_image_src_descriptors[i].nStep = pitch;
        host_image_dst_descriptors[i].pData = nppiMalloc_8u_C3(MAX_SRC_SIZE_W, MAX_SRC_SIZE_H, &pitch);
        host_image_dst_descriptors[i].nStep = pitch;
    }

    return 0;
}

int BatchGpuResize::AddItem(void* src_data_ptr, int src_width, int src_hight, int src_step,
                            void* dst_data_ptr, int dst_width, int dst_hight, int dst_step,
                            ImgType img_type) {
    assert(batch_size < max_batch_size);

    int elem_size = GetElemSize(img_type);
    host_image_src_descriptors[batch_size].oSize = {.width = src_width, .height = src_hight};
    host_image_dst_descriptors[batch_size].oSize = {.width = dst_width, .height = dst_hight};
    host_roi_descriptors[batch_size].oSrcRectROI = {.x = 0, .y = 0, .width = src_width, .height = src_hight};
    host_roi_descriptors[batch_size].oDstRectROI = {.x = 0, .y = 0, .width = dst_width, .height = dst_hight};

    CHECK(cudaMemcpy2D(host_image_src_descriptors[batch_size].pData, host_image_src_descriptors[batch_size].nStep,
                       src_data_ptr,                        src_step,
                       src_width * elem_size,               src_hight,
                       cudaMemcpyHostToDevice));

    host_dst_mat[batch_size].data  = dst_data_ptr;
    host_dst_mat[batch_size].width = dst_width;
    host_dst_mat[batch_size].hight = dst_hight;
    host_dst_mat[batch_size].step  = dst_step;
    batch_size = batch_size + 1;

}

int BatchGpuResize::DoResize(ImgType img_type,  NppiInterpolationMode resize_type) {

    int elem_size = GetElemSize(img_type);

    CHECK(cudaMemcpy(dev_image_src_descriptors ,
                     host_image_src_descriptors,
                     batch_size * sizeof(NppiImageDescriptor),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dev_image_dst_descriptors,
                     host_image_dst_descriptors,
                     batch_size * sizeof(NppiImageDescriptor),
                     cudaMemcpyHostToDevice));
                     
                                          
    CHECK(cudaMemcpy(dev_roi_descriptors,
                     host_roi_descriptors,
                     batch_size * sizeof(NppiResizeBatchROI_Advanced),
                     cudaMemcpyHostToDevice));
                      
    NppStatus result;
    if ( img_type == U8C3) {
         result = nppiResizeBatch_8u_C3R_Advanced(MAX_SRC_SIZE_W,            MAX_SRC_SIZE_H,
                                                  dev_image_src_descriptors, dev_image_dst_descriptors,
                                                  dev_roi_descriptors,       batch_size,
                                                  resize_type);
        if (result != NPP_SUCCESS)
        {
            std::cerr << "Error executing NPP Batch Resize -- code: " << result << std::endl;
        }
    }
    else if (img_type == U8C4) {
         result = nppiResizeBatch_8u_C4R_Advanced(MAX_SRC_SIZE_W,            MAX_SRC_SIZE_H,
                                                  dev_image_src_descriptors, dev_image_dst_descriptors,
                                                  dev_roi_descriptors,       batch_size,
                                                  resize_type);
        if (result != NPP_SUCCESS)
        {
            std::cerr << "Error executing NPP Batch Resize -- code: " << result << std::endl;
        }
    }
    else {
        assert(false);
    }
       
    for (int i = 0; i < batch_size; i++)
    {
        CHECK(cudaMemcpy2D(host_dst_mat[i].data,                 host_dst_mat[i].step,
                           host_image_dst_descriptors[i].pData,  host_image_dst_descriptors[i].nStep, 
                           host_dst_mat[i].width * elem_size,    host_dst_mat[i].hight,
                           cudaMemcpyDeviceToHost));
        
    }
    
    return result;
}


int main()
{
    #define test_batch 8//npp batch 接口最小batch size是2,,如果只有一张图是会error的
    #define max_batch 8
    #define RES_W 512
    #define RES_H 512
    vector <Mat> src_mat_v;
    vector <Mat> dst_mat_v;
    
    for (int i = 0; i < test_batch; i++)
    {
        char img_name [128] = {0};
        sprintf(img_name, "/home/shankun.shankunwan/work/dali/images/sc/sc_%d.jpg", i);
        Mat src_mat;
        src_mat = imread(img_name, CV_LOAD_IMAGE_COLOR);
        Mat dst_mat(RES_W, RES_H, CV_8UC3);
        
        src_mat_v.push_back(src_mat);
        dst_mat_v.push_back(dst_mat);
    }
    
    BatchGpuResize test;
    test.Init(max_batch);

    for (int i = 0; i < test_batch; i++)
    {
        test.AddItem(src_mat_v[i].data, src_mat_v[i].cols, src_mat_v[i].rows, src_mat_v[i].step,
                     dst_mat_v[i].data, dst_mat_v[i].cols, dst_mat_v[i].rows, dst_mat_v[i].step,
                     U8C3);
    }
    
    test.DoResize(U8C3, NPPI_INTER_SUPER);
    test.Destroy();

    for (int i = 0; i < test_batch; i++)
    {
        char img_name [128] = {0};
        sprintf(img_name, "/home/shankun.shankunwan/work/dali/images/sc/ot_%d.bmp", i);
        imwrite(img_name, dst_mat_v[i]);
    }
    
    return 0;
}


