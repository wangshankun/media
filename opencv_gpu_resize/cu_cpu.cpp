#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gapi/gpu/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"
#include<time.h>
//Gpu mat 定义
//https://github.com/opencv/opencv/blob/master/modules/core/src/cuda/gpu_mat.cu

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main()
{

    double t1 = 0;

    cv::cuda::Stream stream;


    struct timespec t_start, t_end;

    
    Mat src = imread("./1.jpg");
    
    if(!src.data){
            cout<<"Image Not Found: "<< endl;
            return -1;
    }
    
    cuda::GpuMat d_src, d_dst;
    Mat dst;
    
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(int i = 0; i < 100; i++)
    {
        d_src.upload(src, stream);
        cuda::resize(d_src, d_dst, Size(512, 512),0 ,0 ,cv::INTER_LINEAR, stream);
        d_dst.download(dst, stream);
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    
    t1 += (t_end.tv_sec  - t_start.tv_sec);
    t1 += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;
    
    cout << "using  time:" << (t1) << "s" << endl;
}
