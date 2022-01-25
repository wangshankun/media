#include<opencv2/opencv.hpp>
#include<iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
using namespace cv;
using namespace std;

vector<string> split(const string& str, const string& delim)
{
    vector<string> res;
    if("" == str) return res;

    char * strs = new char[str.length() + 1] ; 
    strcpy(strs, str.c_str()); 
 
    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());
 
    char *p = strtok(strs, d);
    while(p) {
        string s = p;
        res.push_back(s);
        p = strtok(NULL, d);
    }
 
    return res;
}


int main() 
{
    int obj_num = 20;
    
    vector<Mat> objs;
    {   
        char fname[64] = {0};
        for(int i = 0; i < obj_num; i++)
        {
            sprintf(fname, "obj_%d.jpg", i);
            Mat obj = imread(fname);
            objs.push_back(obj);
        }
    }

    vector<Rect> obj_boxs;
    {
        string line;
        ifstream rect_file ("box_rect_idx.txt");
        if (rect_file.is_open())
        {
             while (getline(rect_file, line))
             {
                vector<string> rect_mbs = split(line, " ");
                Rect rect (stoi(rect_mbs[0]), stoi(rect_mbs[1]), stoi(rect_mbs[2]), stoi(rect_mbs[3]));
                obj_boxs.push_back(rect);
             }
             rect_file.close();
        }
    }
    
    vector<Mat> obj_masks;
    {   
        char fname[64] = {0};
        for(int i = 0; i < obj_num; i++)
        {
            sprintf(fname, "mask_obj_%d.jpg", i);
            Mat mask_obj = imread(fname,0);
            //cv::threshold(mask_obj, mask_obj, 0, 255, THRESH_BINARY);
            obj_masks.push_back(mask_obj);
        }
    }
    
    Mat bg = imread("bg.jpg");
    Mat result = bg.clone();
    
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i =0 ; i < 100; i++)
    {
        Mat weight_mask(result.size(), CV_8UC1, Scalar::all(0));
        {
            for(int i = 0; i < obj_num; i++)
            {
                weight_mask(obj_boxs[i]) = weight_mask(obj_boxs[i]) +  obj_masks[i] / 255;
            }
        }
        
        cv::Mat weight_mask_bgr, whole_mask_bgr, whole_mask_bgr_nor;
        cv::cvtColor(weight_mask, weight_mask_bgr, COLOR_GRAY2BGR);
        cv::threshold(weight_mask_bgr, whole_mask_bgr, 0, 255, THRESH_BINARY);
        cv::bitwise_not(whole_mask_bgr, whole_mask_bgr_nor);
        whole_mask_bgr_nor.copyTo(result, whole_mask_bgr);

        {   
            for(int i = 0; i < obj_num; i++)
            {
                Mat out_obj, obj_with_bg, obj_masks_bgr;
                addWeighted(objs[i], 0.2, bg(obj_boxs[i]), (1.0 - 0.2), 0, obj_with_bg);
                cv::cvtColor(obj_masks[i], obj_masks_bgr, COLOR_GRAY2BGR);
                cv::bitwise_and(obj_with_bg, obj_masks_bgr, out_obj);
                
                out_obj = out_obj / weight_mask_bgr(obj_boxs[i]);
                result(obj_boxs[i]) = result(obj_boxs[i]) +  out_obj;
            }
        }

    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("ave elapsed time:%f\r\n",elapsed/100);

    imwrite("result.jpg",result);
    imshow("result", result);
    waitKey();
    return 0;
}

