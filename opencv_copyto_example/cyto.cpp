
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include<iostream>
using namespace cv;
 
bool ROI_AddImage()
{
    Mat bg   = imread("bg.jpg");
    Mat mask_bgr = imread("mask_obj_0.jpg");//原本是二值图，但是jpg反解压出来会有毛刺,如果存储原始位图就没有这个问题
    //cv::threshold(mask_bgr, mask_bgr, 0, 255, THRESH_BINARY);
    mask_bgr.copyTo(bg(Rect(300, 100, mask_bgr.cols, mask_bgr.rows)), mask_bgr);
    
    namedWindow("效果图");
    imshow("效果图", bg);
    
}
int main()
{
    ROI_AddImage();
    waitKey();
    return 0;
}
