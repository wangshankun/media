#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>


extern "C"
{
extern void compress_2_h265(unsigned char* in_datas, int* sizes, int in_num, int width, int height, unsigned char* out_data, int *out_size);
}

using namespace std;

extern int readInput(const std::string &sInputPath,
              std::vector<std::string> &filelist);

typedef vector<char>  FileData;

int main()
{
    
    vector<string> inputFiles;
    if (readInput("/home/shankun.shankunwan/si112/", inputFiles))
    {
        cerr << "Cannot open path dir: ";
    }

    
    printf("inputFiles.size():%d \r\n",inputFiles.size());
    
    FileData                   file_datas;
    vector< int>               file_sizes;
    
    double elapsed = 0;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (unsigned int i = 0; i < inputFiles.size(); i++)
    {
        string &sFileName = inputFiles[i];
        FileData raw_data;
        std::ifstream input(sFileName, std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
            cerr << "Cannot open image: ";
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
    compress_2_h265((unsigned char*)(file_datas.data()), file_sizes.data(), file_sizes.size(), 1920, 1080,  out_data,  &out_size);
    printf("out_size: %d \r\n",out_size);
    free(out_data);

    return 0;
}
