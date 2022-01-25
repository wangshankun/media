#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
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

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path);
  image.convertTo(image, CV_32FC3);
  cv::resize(image, image, cv::Size(1024, 1024));

  //image.convertTo(image, CV_32FC3, 1.0/255*3.2, -1.6);
  return image;
}

__global__ static void setvalue_gpu_kernel(float *data,size_t size,float value)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    if(tid<size)
    {
        data[tid] = value;
    }
}


void setvalue_gpu(float *data, size_t size,float value)
{
    int blockNum = (size+255) / 256;
    setvalue_gpu_kernel<<<blockNum,256,0>>>(data,size,value);
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: conv <image> [gpu=0]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;


  cudaSetDevice(gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/1024,
                                        /*image_width=*/1024));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/1024,
                                        /*image_width=*/1024));

  int image_bytes = 1*3*1024*1024*sizeof(float);
  int image_sizes = 1*3*1024*1024;
  float alpha = 1.0/255.0*3.2;
  float beta  = 1;
  //float alpha = 1;float beta = 0;

  float* h_output = new float[3*1024*1024];

  double elapsed;
  struct timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  cv::Mat image = load_image(argv[1]);

  float* d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);


  float* d_output{nullptr};
  cudaMalloc(&d_output, image_bytes);
  setvalue_gpu(d_output, image_sizes, -1.6);

  //img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
  //nwhc -> nchw; scale; mean
  checkCUDNN(cudnnTransformTensor(
                                    cudnn,
                                    &alpha,
                                    input_descriptor,
                                    d_input,
                                    &beta,
                                    output_descriptor,
                                    d_output))

  //cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("elapsed time:%f\r\n",elapsed);

//  savefile("cvpp.bin", h_output, image_bytes);

  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);

  cudnnDestroy(cudnn);
}
