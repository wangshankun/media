#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <jpeglib.h>
#include <turbojpeg.h>
#include <cstring>

void yuv420sp_to_yuv420p(unsigned char* yuv420sp, unsigned char* yuv420p, int width, int height)
{
    int i, j;
    int y_size = width * height;

    unsigned char* y = yuv420sp;
    unsigned char* uv = yuv420sp + y_size;

    unsigned char* y_tmp = yuv420p;
    unsigned char* u_tmp = yuv420p + y_size;
    unsigned char* v_tmp = yuv420p + y_size * 5 / 4;

    // y
    memcpy(y_tmp, y, y_size);

    // u
    for (j = 0, i = 0; j < y_size/2; j+=2, i++)
    {
        u_tmp[i] = uv[j];
        v_tmp[i] = uv[j+1];
    }
}


void yuv420p_to_yuv420sp(unsigned char* yuv420p, unsigned char* yuv420sp, int width, int height)
{
        int i, j;
        int y_size = width * height;
        int y_begin = 0;
        int u_begin = y_size;
        int v_begin = y_size * 5 / 4;
        int y_temp = 0;
        int uv_temp = y_size;
        // y
        for(i = 0;i < y_size; i++){
            yuv420sp[i] = yuv420p[i];
        }

        // u
        for(j = 0, i = 0; j < y_size / 2; j+=2, i++){
            // nv 21
            yuv420sp[uv_temp + j] = yuv420p[v_begin + i];
            yuv420sp[uv_temp + j + 1] = yuv420p[u_begin + i];
        }
}

int tjpeg2bgr(const char* jpeg_name, unsigned char** bgr_buffer, int* bgr_size, int* width, int* height)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE * infile;
    if ((infile = fopen(jpeg_name, "rb")) == NULL)
    {
        fprintf(stderr, "can't open %s\n", jpeg_name);
        return -1;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_EXT_BGR;
    jpeg_start_decompress(&cinfo);

    *width    = cinfo.output_width;
    *height   = cinfo.output_height;
    *bgr_size = *width * *height * cinfo.output_components;
    *bgr_buffer =(unsigned char *)malloc(*bgr_size);
    if (*bgr_buffer == NULL)
    {
        printf("malloc buffer for rgb failed.\n");
        return -1;
    }

    int row_stride = cinfo.output_width * cinfo.output_components;
    JSAMPARRAY buffer = (JSAMPARRAY)malloc(sizeof(JSAMPROW));
    buffer[0] = (JSAMPROW)malloc(sizeof(JSAMPLE) * row_stride);
    long counter = 0;
    while (cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(*bgr_buffer + counter, buffer[0], row_stride);
        counter += row_stride;
    }
    free(buffer);

    fclose(infile);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}


int tjpeg2yuv(unsigned char* jpeg_buffer, int jpeg_size, unsigned char** yuv_buffer, int* yuv_size, int* yuv_type, int *w, int *h)
{
    tjhandle handle = NULL;
    int width, height, subsample, colorspace;
    int flags = 0;
    int padding = 1; // 1或4均可，但不能是0
    int ret = 0;

    handle = tjInitDecompress();
    tjDecompressHeader3(handle, jpeg_buffer, jpeg_size, &width, &height, &subsample, &colorspace);
    *w = width;
    *h = height;
    //printf("w: %d h: %d subsample: %d color: %d\n", width, height, subsample, colorspace);

    flags |= 0;

    *yuv_type = subsample;
    // 注：经测试，指定的yuv采样格式只对YUV缓冲区大小有影响，实际上还是按JPEG本身的YUV格式来转换的
    *yuv_size = tjBufSizeYUV2(width, padding, height, subsample);
    *yuv_buffer =(unsigned char *)malloc(*yuv_size);
    if (*yuv_buffer == NULL)
    {
        printf("malloc buffer for rgb failed.\n");
        return -1;
    }
    ret = tjDecompressToYUV2(handle, jpeg_buffer, jpeg_size, *yuv_buffer, width,
                             padding, height, flags);
    if (ret < 0)
    {
        printf("compress to jpeg failed: %s\n", tjGetErrorStr());
    }
    tjDestroy(handle);

    return ret;
}


int main(int argc, char* argv[]) {
    printf("↓↓↓↓↓↓↓↓↓↓ Decode JPEG to BGR ↓↓↓↓↓↓↓↓↓↓\n");

    uint8_t *bgr_data;
    int bgr_size;
    int bgr_width;
    int bgr_height;
        
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < 100; i++)
    {
        char *inJpegName3 = (char*)"test.jpg";
        tjpeg2bgr(inJpegName3, &bgr_data, &bgr_size, &bgr_width, &bgr_height);
        
        if(i < 99) free(bgr_data);//最后一次不在循环里面释放
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("per jpeg decode elapsed time:%f\r\n", elapsed/100);
    
    printf("size: %d width:%d height:%d\n", bgr_size, bgr_width, bgr_height);
    FILE *bgr_file = fopen("test.bgr", "wb");
    fwrite(bgr_data, bgr_size, 1, bgr_file);
    
    free(bgr_data);
    fflush(bgr_file);
    fclose(bgr_file);
    printf("↑↑↑↑↑↑↑↑↑↑ Decode JPEG to BGR ↑↑↑↑↑↑↑↑↑↑\n\n");
}


/*
int main(int argc, char* argv[]) {
    printf("↓↓↓↓↓↓↓↓↓↓ Decode JPEG to YUV ↓↓↓↓↓↓↓↓↓↓\n");

    uint8_t *yuv_nv21;
    int yuvSize;
    int yuvType;
    int w;
    int h;
        
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < 100; i++)
    {
        char *inJpegName3 = "test.jpg";
        FILE *jpegFile = fopen(inJpegName3, "rb");
        
        struct stat statbuf;
        stat(inJpegName3, &statbuf);
        int fileLen=statbuf.st_size;
        //printf("fileLength: %d\n", fileLen);

        uint8_t *jpegData = (uint8_t*)malloc(fileLen);
        fread(jpegData, fileLen, 1, jpegFile);
        fclose(jpegFile);
        
        uint8_t *yuvData;
        tjpeg2yuv(jpegData, fileLen, &yuvData, &yuvSize, &yuvType, &w, &h);
        free(jpegData);
        
        yuv_nv21 = (uint8_t*)malloc(yuvSize);
        yuv420p_to_yuv420sp(yuvData, yuv_nv21, w,  h );
        free(yuv_nv21);
        if(i < 99) free(yuv_nv21);//最后一次不在循环里面释放
        free(yuvData);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("per jpeg decode elapsed time:%f\r\n", elapsed/100);
    
    printf("size: %d; type: %d\n", yuvSize, yuvType);

    FILE *yuvFile = fopen("test.nv21", "wb");
    fwrite(yuv_nv21, yuvSize, 1, yuvFile);
    free(yuv_nv21);
    fflush(yuvFile);
    fclose(yuvFile);
    printf("↑↑↑↑↑↑↑↑↑↑ Decode JPEG to YUV ↑↑↑↑↑↑↑↑↑↑\n\n");
}
*/
