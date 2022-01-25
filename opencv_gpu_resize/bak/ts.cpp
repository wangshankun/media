#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gapi/gpu/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"
#include<time.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;




int main(int argc, char *argv[])
{
    // 读取图片为 8bit单通道 Mat
    cv::Mat rawImage = cv::imread("test.jpg", CV_8UC1);

    // 申请Page-locked内存 allocate page-locked memory
    cv::cuda::CudaMem host_src_pl(rawImage.cols, rawImage.rows, CV_8UC1, cv::cuda::CudaMem::ALLOC_PAGE_LOCKED);
    cv::cuda::CudaMem host_dst_pl;

    // 填满申请到的PL内存 Do something to fill host_src_pl
    memcpy(host_src_pl.datastart, rawImage.datastart, host_src_pl.cols * host_src_pl.rows);

    // 创建GpuMat
    cv::cuda::GpuMat gpu_src,gpu_dst;

    // 创建Stream
    cv::cuda::Stream stream;
    
    // 从主机（CPU）向设备（GPU）复制数据
    stream.enqueueUpload(host_src_pl, gpu_src);

    // 在设备中对数据进行处理
    cv::cuda::threshold(gpu_src, gpu_dst, 128.0, 255.0, CV_THRESH_BINARY, stream);

    // 从设备向主机复制数据
    stream.enqueueDownload(gpu_dst, host_dst_pl);

    // 在主机中与设备并行的进行其他的处理
    cv::Mat binImage;
    cv::threshold(rawImage, binImage, 128.0, 255.0, CV_THRESH_BINARY);

    // 等待设备完成计算任务
    stream.waitForCompletion();

    // 现在可以使用设备中计算得到的数据
    cv::Mat host_dst = host_dst_pl;

    cv::imwrite("host_dst.jpg", host_dst);
    cv::imwrite("binImage.jpg", binImage);

    return 0;
}
