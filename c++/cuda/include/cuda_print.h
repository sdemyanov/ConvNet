/*
Copyright (C) 2016 Sergey Demyanov.
contact: sergey@demyanov.net
http://www.demyanov.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _CUDA_PRINT_H_
#define _CUDA_PRINT_H_

#include "mex_print.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

const char* curandGetErrorString(curandStatus_t error);
const char* cublasGetErrorString(cublasStatus_t status);

#ifndef CUDA_CALL
  #define CUDA_CALL(fun) { \
    cudaError_t status = (fun); \
    if (status != cudaSuccess) { \
      mexAssertMsg(false, cudaGetErrorString(status)); \
    } \
  }
#endif

#ifndef CURAND_CALL
  #define CURAND_CALL(fun) { \
    curandStatus_t status = (fun); \
    if (status != CURAND_STATUS_SUCCESS) { \
      mexAssertMsg(false, curandGetErrorString(status)); \
    } \
  }
#endif

#ifndef CUBLAS_CALL
  #define CUBLAS_CALL(fun) { \
    cublasStatus_t status = (fun); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      mexAssertMsg(false, cublasGetErrorString(status)); \
    } \
  }
#endif

#ifndef CUDNN_CALL
  #define CUDNN_CALL(fun) { \
    cudnnStatus_t status = (fun); \
    if (status != CUDNN_STATUS_SUCCESS) { \
      mexAssertMsg(false, cudnnGetErrorString(status)); \
    } \
  }
#endif

#endif
