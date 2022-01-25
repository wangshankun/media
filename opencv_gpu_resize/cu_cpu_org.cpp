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

int main()
{
    int k = 100;
    double t1 = 0;
    double t2 = 0;
    double t3 = 0;
    double t4 = 0;
    double t5 = 0;
    
    cv::cuda::Stream stream;

    while (k > 0)
    {

        struct timespec t_start, t_end_read, t_end_h2d, t_end_resize, t_end_d2h, t_end_cpu;

        clock_gettime(CLOCK_MONOTONIC, &t_start);
        Mat src = imread("./test.jpg");
        clock_gettime(CLOCK_MONOTONIC, &t_end_read);
        if(!src.data){
                cout<<"Image Not Found: "<< endl;
                return -1;
        }

        cuda::GpuMat d_src;
        d_src.upload(src, stream);
        clock_gettime(CLOCK_MONOTONIC, &t_end_h2d);
        cuda::GpuMat d_dst;

        cuda::resize(d_src, d_dst, Size(512, 512),0 ,0 ,cv::INTER_LINEAR, stream);
        clock_gettime(CLOCK_MONOTONIC, &t_end_resize);

        Mat dst;
        d_dst.download(dst, stream);
        clock_gettime(CLOCK_MONOTONIC, &t_end_d2h);
        
        cv::resize(src, dst, Size(512, 512));
        clock_gettime(CLOCK_MONOTONIC, &t_end_cpu);

        t1 += (t_end_read.tv_sec    - t_start.tv_sec);
        t1 += (t_end_read.tv_nsec   - t_start.tv_nsec) / 1000000000.0;
        t2 += (t_end_h2d.tv_sec     - t_end_read.tv_sec);
        t2 += (t_end_h2d.tv_nsec    - t_end_read.tv_nsec) / 1000000000.0;
        t3 += (t_end_resize.tv_sec  - t_end_h2d.tv_sec);
        t3 += (t_end_resize.tv_nsec - t_end_h2d.tv_nsec) / 1000000000.0;
        t4 += (t_end_d2h.tv_sec     - t_end_resize.tv_sec);
        t4 += (t_end_d2h.tv_nsec    - t_end_resize.tv_nsec) / 1000000000.0;
        t5 += (t_end_cpu.tv_sec     - t_end_d2h.tv_sec);
        t5 += (t_end_cpu.tv_nsec    - t_end_d2h.tv_nsec) / 1000000000.0;
        k = k - 1;
    }
    /*
    cout << "read       time:" << chrono::duration<double, milli>(t_end_read   - t_start     ).count() << "ms" << endl;
    cout << "upload     time:" << chrono::duration<double, milli>(t_end_h2d    - t_end_read  ).count() << "ms" << endl;
    cout << "gpu resize time:" << chrono::duration<double, milli>(t_end_resize - t_end_h2d   ).count() << "ms" << endl;
    cout << "download   time:" << chrono::duration<double, milli>(t_end_d2h    - t_end_resize).count() << "ms" << endl;
    cout << "cpu resize time:" << chrono::duration<double, milli>(t_end_cpu    - t_end_d2h   ).count() << "ms" << endl;
    */
    cout << "read       time:" << (t1) << "s" << endl;
    cout << "upload     time:" << (t2) << "s" << endl;
    cout << "gpu resize time:" << (t3) << "s" << endl;
    cout << "download   time:" << (t4) << "s" << endl;
    cout << "cpu resize time:" << (t5) << "s" << endl;
}
