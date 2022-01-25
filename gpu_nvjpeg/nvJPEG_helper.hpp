/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This sample needs at least CUDA 10.0.
// It demonstrates usages of the nvJPEG library

#ifndef NV_JPEG_EXAMPLE
#define NV_JPEG_EXAMPLE

#include "cuda_runtime.h"
#include "nvjpeg.h"
#include "helper_cuda.h"
#include "helper_timer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <string.h>    // strcmpi
#include <sys/time.h>  // timings

#include <dirent.h>  // linux dir traverse
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// write bmp, input - RGB, device
int writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
             const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height); 

// write bmp, input - RGB, device
int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
              int width, int height);

int readInput(const std::string &sInputPath,
              std::vector<std::string> &filelist);

int writeBMPi_test(const char *filename, std::vector<unsigned char> *vchanRGB, int pitch,
              int width, int height);
#endif
