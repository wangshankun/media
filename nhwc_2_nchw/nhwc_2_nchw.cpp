#include <opencv2/opencv.hpp>
#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

int main()
{
    cv::Mat image = cv::imread("test.jpg");
    image.convertTo(image, CV_32FC3);
    cv::resize(image, image, cv::Size(1024, 1024));

    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    //image = image / 255.0 * 3.2 - cv::Scalar(1.6f, 1.6f, 1.6f);
    image = image / 255.0 * 3.2 - 1.6f;
    float* dst = new float[1024*1024*3];
    float* src = (float*)(image.data);
    for(int i = 0; i < 1024*1024; i++)
    {
        dst[i] = src[i * 3];
        dst[i + 1024*1024] = src[i * 3 + 1];
        dst[i + 1024*1024*2] = src[i * 3 + 2];
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);



    elapsed = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < 1024*1024; i++)
    {
        dst[i] = src[i * 3]/255*3.2 - 1.6;
        dst[i + 1024*1024] = src[i * 3 + 1]/255*3.2 - 1.6;
        dst[i + 1024*1024*2] = src[i * 3 + 2]/255*3.2 - 1.6;
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);

//    savefile("cvpp.bin",dst,3*1024*1024*sizeof(float));
}
