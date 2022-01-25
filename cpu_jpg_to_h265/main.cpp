#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


static void print_hex_str(const void* buf , size_t size)
{
    unsigned char* str = (unsigned char*)buf;
    char line[512] = {0};
    const size_t lineLength = 16; // 8或者32
    char text[24] = {0};
    char* pc;
    int textLength = lineLength;
    size_t ix = 0 ;
    size_t jx = 0 ;

    for (ix = 0 ; ix < size ; ix += lineLength) {
        sprintf(line, "%.8xh: ", ix);
// 打印16进制
        for (jx = 0 ; jx != lineLength ; jx++) {
            if (ix + jx >= size) {
                sprintf(line + (11 + jx * 3), "   "); // 处理最后一行空白
                if (ix + jx == size)
                    textLength = jx;  // 处理最后一行文本截断
            } else
                sprintf(line + (11 + jx * 3), "%.2X ", * (str + ix + jx));
        }
// 打印字符串
        {
            memcpy(text, str + ix, lineLength);
            pc = text;
            while (pc != text + lineLength) {
                if ((unsigned char)*pc < 0x20) // 空格之前为控制码
                    *pc = '.';                 // 控制码转成'.'显示
                pc++;
            }
            text[textLength] = '\0';
            sprintf(line + (11 + lineLength * 3), "; %s", text);
        }

        printf("%s\n", line);
    }
}


extern "C"
{
extern int compress_2_h265_cpu(unsigned char* in_datas, int* sizes, int in_num, unsigned char** out_data, int *out_size);
}

using namespace std;

typedef vector<char>  FileData;

int main()
{
    
    vector<string> inputFiles;
    
    FileData                   file_datas;
    vector< int>               file_sizes;
    
    for(int i = 1; i < 96; i++)
    {
        char name[128] = {0};
        sprintf(name, "/home/shankun.shankunwan/work/si112/%04d.jpg", i);
        FileData raw_data;
        std::ifstream input(name, std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
            cerr << "Cannot open image: " << name;
        }
        // Get the size
        streamsize file_size = input.tellg();
        input.seekg(0, ios::beg);
        // resize if buffer is too small
        if (raw_data.size() < file_size)
        {
            raw_data.resize(file_size);
        }

        if (!input.read(raw_data.data(), file_size))
        {
            cerr << "Cannot read from file: ";
        }
        
        file_datas.reserve(file_datas.size() + raw_data.size());
        file_datas.insert(file_datas.end(), raw_data.begin(), raw_data.end());

        file_sizes.push_back(int(file_size));
    }
    
    int   out_size = 0;
    unsigned char* out_data;
    printf("begin comprss\r\n");
    double time_escp = 0;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    compress_2_h265_cpu((unsigned char*)(file_datas.data()), file_sizes.data(), file_sizes.size(), &out_data,  &out_size);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    time_escp += (t_end.tv_sec  - t_start.tv_sec);
    time_escp += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;
    printf("use time:%f \r\n",time_escp);

    print_hex_str(out_data,64);
    savefile("test.hevc", out_data, out_size);
    free(out_data);

    return 0;
}
