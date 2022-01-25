#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

int get_dir_list(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        if(dirp->d_type = 8 && dirp->d_reclen == 32) files.push_back(string(dirp->d_name));//判断文件类型
        //printf(" d_name :%s  d_reclen:%d  d_type:%d\r\n", dirp->d_name, dirp->d_reclen, dirp->d_type);
    }
    closedir(dp);
    return 0;
}

int get_file_size(const char* file_name)
{
    struct stat statbuf;

    if (stat(file_name, &statbuf) == -1) 
    {
      return -1;
    }

    return statbuf.st_size;
}

typedef class CPos{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar & s_idx;
            ar & e_idx;
        }
    public:
        uint16_t s_idx;
        uint16_t e_idx;
} Pos_t;

typedef std::vector<Pos_t> Pos_vec;

typedef class CBin_pic {
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar & rows;
            ar & cols;
            ar & rows_record;
        }
        
    public:
        uint16_t rows;
        uint16_t cols;
        std::vector<Pos_vec> rows_record;
} Bin_pic_t;

static void binary_picture_encode(cv::Mat bgr_mat,  Bin_pic_t* en_pic)
{
    if ( ! bgr_mat.isContinuous() )
    { 
        bgr_mat = bgr_mat.clone();
    }

    en_pic->rows = (uint16_t)(bgr_mat.rows);
    en_pic->cols = (uint16_t)(bgr_mat.cols);

    for (uint16_t j = 0; j < bgr_mat.rows; j++)
    {
        uint8_t  *data = bgr_mat.ptr<uint8_t>(j);
        Pos_vec row_record_list;
        uint8_t  t_f = 0; //记录开关
        uint16_t i = 0;
        Pos_t pos_elem = {0, 0};
        
        for (i = 0; bgr_mat.cols - i >= 8; )//确保不溢出
        {
                if (*((uint64_t*)(data + i)) != 0 &&  t_f == 0)//uint64_t，8个一次做判断，效率两倍于uint8_t
                {
                    for(uint8_t x = 0; x < 8; x++)
                    {
                        if(*(data + i+ x) == 0)
                        {
                            i = i + 1;
                        }
                        else
                        {
                            pos_elem.s_idx = i;
                            t_f = 1;
                            break;
                        }
                    }
                }
                else if(*((uint64_t*)(data + i)) == 0 &&  t_f == 1)
                {
                    i = i - 8;//前移8个记录再统计从哪个字节开始为0
                    for(uint8_t x = 0; x < 8; x++)
                    {
                        if(*(data + i + x) != 0)
                        {
                            i = i + 1;
                        }
                        else
                        {
                             pos_elem.e_idx = i;
                             row_record_list.push_back(pos_elem);
                             t_f = 0;
                             break;
                        }
                    }
                }
                i = i + 8;
        }

        for (; i < bgr_mat.cols; i++)//不足8个的按单个字节做
        {
            if(*(data + i) == 0 &&  t_f == 1)
            {
                 pos_elem.e_idx = i;
                 row_record_list.push_back(pos_elem);
                 t_f = 0;
            }
            else if(*(data + i) != 0 &&  t_f == 0)
            {
                pos_elem.s_idx = i;
                t_f = 1;
            }
        }

        if(t_f == 1) //如果到行尾还再记录，那么结束时候加上节点
        {
             pos_elem.e_idx = bgr_mat.cols - 1;
             row_record_list.push_back(pos_elem);
             t_f = 0;
        }
            
        if(row_record_list.size() == 0) //如果没有记录默认填0,为了行占位
        {
            row_record_list.push_back(pos_elem);
        }

        en_pic->rows_record.push_back(row_record_list);
    }
}

static void binary_picture_decode(cv::Mat &bgr_mat, Bin_pic_t de_pic)
{
    
    bgr_mat = cv::Mat::zeros(de_pic.rows, de_pic.cols, CV_8UC1);

    for (int j = 0; j < de_pic.rows_record.size(); j++)
    {
        uint8_t  *data = bgr_mat.ptr<uint8_t>(j);
        for(auto &pos_elem: de_pic.rows_record[j])
        {
            memset(data + pos_elem.s_idx, 255, pos_elem.e_idx - pos_elem.s_idx);
        }
    }
}

int main()
{
    double elapsed;
    struct timespec start, finish;

    string dir = string("./test/");
    vector<string> files = vector<string>();
    get_dir_list(dir, files);

    clock_gettime(CLOCK_MONOTONIC, &start);
    vector<cv::Mat> test_png;
    for (auto &d: files) 
    {
        string full_file_name = dir + d;
        cv::Mat test_img = cv::imread(full_file_name, CV_LOAD_IMAGE_GRAYSCALE);
        test_png.push_back(test_img);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("test_png read elapsed time:%f\r\n",elapsed);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto &d: test_png)
    {
        cv::imwrite("tmp.png",d);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("test_png write elapsed time:%f\r\n",elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);

    std::vector<Bin_pic_t> ser_bin_pics;
    //编码
    for (auto &test_img: test_png) 
    {
        Bin_pic_t en_pic;
        binary_picture_encode(test_img, &en_pic);
        ser_bin_pics.push_back(en_pic);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("binary_picture_encode elapsed time:%f\r\n",elapsed);

    printf("%d %d\r\n",sizeof(Bin_pic_t) , ser_bin_pics.size());

    //序列化
    clock_gettime(CLOCK_MONOTONIC, &start);
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << ser_bin_pics;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("ser_encode elapsed time:%f\r\n",elapsed);
    printf("ss.size:%d  \r\n", ss.str().size());
    //反序列化
    clock_gettime(CLOCK_MONOTONIC, &start);
    std::vector<Bin_pic_t> rser_bin_pics;
    boost::archive::binary_iarchive ia(ss);
    ia >> rser_bin_pics;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("rser_encode elapsed time:%f\r\n",elapsed);
    
    //测试一下
    cv::Mat bgr_mat;
    binary_picture_decode(bgr_mat, rser_bin_pics[0]);
    cv::imwrite("./x.bmp",bgr_mat);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    //解码
    for (auto &bin_pic: rser_bin_pics) 
    {
        cv::Mat bgr_mat;
        binary_picture_decode(bgr_mat, bin_pic);
        //cv::imwrite("./x.bmp",bgr_mat);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("binary_picture_decode elapsed time:%f\r\n",elapsed);

    return 0;
}
