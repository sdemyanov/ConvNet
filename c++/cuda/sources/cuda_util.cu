/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_util.h"
#include <cuda.h>

#define MUL24(x, y) ((x) * (y))

#define WARP_SIZE                           32

#define ELTWISE_FLAT_THREADS_X              128
#define ELTWISE_THREADS_X                   32
#define ELTWISE_THREADS_Y                   8
#define ADD_VEC_THREADS_X                   64
#define ADD_VEC_THREADS_Y                   4
#define NUM_BLOCKS_MAX                      65535
#define NUM_SUM_COLS_THREADS_PER_BLOCK      128
#define AGG_SHORT_ROWS_THREADS_X            32
#define AGG_SHORT_ROWS_THREADS_Y            8
#define AGG_SHORT_ROWS_LOOPS_Y              32
#define AWR_NUM_THREADS                     256
#define AWR_LOG_NUM_WARPS                   3
#define LOG_WARP_SIZE                       5
#define AWR_NUM_WARPS                       AWR_NUM_THREADS / WARP_SIZE 
#define LOGREG_GRAD_THREADS_X               32
#define LOGREG_GRAD_THREADS_Y               4
#define DP_BLOCKSIZE                        512

// device

template<typename T> 
__device__ T shfl_down(T a, int b, int c = WARP_SIZE) {
#if __CUDA_ARCH__ >= 300
    return __shfl_down(a, b, c);
#else
    return 0;
#endif
}

__device__
__constant__ float kDevEps = (float) PRECISION_EPS;

// global

class UnaryOp {
public:
  class Identity {
  public:
    __device__ inline float operator()(const float a) const {
      return a;
    }
  };
  
  class Validate {
  public:
    __device__ inline float operator()(const float a) const {
      if (-kDevEps < a && a < kDevEps) return 0;
      return a;
    }
  };  
  
  class Sign {
  public:
    __device__ inline float operator()(const float a) const {
      return (a > kDevEps) - (a < -kDevEps);
    }
  };
  
  class Sqrt {
  public:
    __device__ inline float operator()(const float a) const {
      return sqrtf(a);
    }
  };
  
  class Log {
  public:
    __device__ inline float operator()(const float a) const {
      return __logf(a);
    }
  };
  
  class Exp {
  public:
    __device__ inline float operator()(const float a) const {
      return __expf(a);
    }
  };
  
  class Sigmoid {
  public:
    __device__ inline float operator()(const float a) const {
      return __fdividef(1.0f, 1.0f + __expf(-a));
    }
  };
  
  class Scalar {
  private:
    const float scalar;
  public:
    Scalar(const float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(const float a) const {
      return scalar;
    }
  };
  
  class AddScalar {
  private:
    const float scalar;
  public:
    AddScalar(const float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(const float a) const {
      return a + scalar;
    }
  };
  
  class MultByScalar {
  private:
    const float scalar;
  public:
    MultByScalar(const float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(const float a) const {
      return a * scalar;
    }
  };
  
  class DivByScalar {
  private:
    const float scalar;
  public:
    DivByScalar(const float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(const float a) const {
      return __fdividef(a, scalar);
    }
  };

  class BiggerThanScalar {
  private:
    const float scalar;
  public:
    BiggerThanScalar(const float _scalar) : scalar(_scalar) {
    }
    __device__ inline float operator()(const float a) const {
      return a > scalar;
    }
  };  
};

class BinaryOp {
public:
    
  class Second {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return b;
    }
  };

  class Add {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return a + b;
    }
  };
  
  class Subtract {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return a - b;
    }
  };
  
  class Multiply {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return a * b;
    }
  };
  
  class Divide {
  public:
    __device__ inline float operator()(const float a, const float b) const  {
      return __fdividef(a, b);
    }
  };
  
  class SigmDer {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return a * b * (1 - b);
    }
  };
};

class Aggs {
public:

  class Sum {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return a + b;
    }
    __device__ inline float getBaseValue() {
      return 0;
    }
  };
  
  class Max {
  public:
    __device__ inline float operator()(const float a, const float b) const {
      return a > b ? a : b;
    }
    __device__ inline float getBaseValue() {
      return -2e38;
    }
  };
  
};

/* ------- Unary operations ------- */

template<class Op>
__global__ void kEltwiseUnaryOpFlat(const float* a, float* const dest, int numElements, Op op) {
    const int idxX = blockIdx.x * ELTWISE_FLAT_THREADS_X + threadIdx.x;
    for (int x = idxX; x < numElements; x += gridDim.x * ELTWISE_FLAT_THREADS_X) {
        dest[x] = op(a[x]);
    }
}

template <class Op> 
void _applyUnaryOp(MatGPU &mat, Op op) {

    if (mat.empty()) return;    
    mexAssert(mat.stride_ == 1, "In _applyUnaryOp stride_ should be 1");    
    int _numElements = (int) (mat.size1_ * mat.size2_);    
    cudaStream_t stream = MatGPU::_defaultStream;    
   
    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
    kEltwiseUnaryOpFlat<Op><<<blocks, threads, 0, stream>>>(mat.data_, mat.data_, _numElements, op);
    mexAssert(cudaGetLastError() == cudaSuccess, "kEltwiseUnaryOpFlat: kernel execution failed");
}

void cuda_validate(MatGPU &mat) {
  _applyUnaryOp(mat, UnaryOp::Validate());
}

void cuda_sign(MatGPU &mat) {
  _applyUnaryOp(mat, UnaryOp::Sign());
}

void cuda_sqrt(MatGPU &mat) {
  _applyUnaryOp(mat, UnaryOp::Sqrt());
}

void cuda_log(MatGPU &mat) {
  _applyUnaryOp(mat, UnaryOp::Log());
}

void cuda_exp(MatGPU &mat) {
  _applyUnaryOp(mat, UnaryOp::Exp());
}

void cuda_sigmoid(MatGPU &mat) {
  _applyUnaryOp(mat, UnaryOp::Sigmoid());
}

/* ------- Unary operations with scalars ------- */

void cuda_assval(MatGPU &mat, float val) {
  _applyUnaryOp(mat, UnaryOp::Scalar(val));  
}

void cuda_addval(MatGPU &mat, float val) {
  _applyUnaryOp(mat, UnaryOp::AddScalar(val));  
}

void cuda_subval(MatGPU &mat, float val) {
  _applyUnaryOp(mat, UnaryOp::AddScalar(-val));  
}

void cuda_multval(MatGPU &mat, float val) {
  _applyUnaryOp(mat, UnaryOp::MultByScalar(val));  
}

void cuda_divval(MatGPU &mat, float val) {
  _applyUnaryOp(mat, UnaryOp::DivByScalar(val));  
}

/* ------- Binary operations ------- */

template<class Op>
__global__ void kEltwiseBinaryOpFlat(const float* a, const float* b, float* const dest, int numElements, Op op) {
    const int idxX = blockIdx.x * ELTWISE_FLAT_THREADS_X + threadIdx.x;
    for (int x = idxX; x < numElements; x += gridDim.x * ELTWISE_FLAT_THREADS_X) {
        dest[x] = op(a[x], b[x]);
    }
}

template <class Op> 
void _applyBinaryOp(MatGPU& mat, const MatGPU& b, Op op) {
    
    if (mat.empty()) return;     
    mexAssert(mat.stride_ == 1 && b.stride_ == 1, "In _applyBinaryOp strides should be 1");    
    mexAssert(mat.order_ == b.order_, "In _applyBinaryOp orders should be the same");    
    mexAssert(mat.size1_ == b.size1_ && mat.size2_ == b.size2_,
      "In _applyBinaryOp the sizes of matrices do not correspond");
    int _numElements = (int) (mat.size1_ * mat.size2_);    
    cudaStream_t stream = MatGPU::_defaultStream;    
    
    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
    kEltwiseBinaryOpFlat<Op><<<blocks, threads, 0, stream>>>(mat.data_, b.data_, mat.data_, _numElements, op);    
    mexAssert(cudaGetLastError() == cudaSuccess, "kEltwiseBinaryOpFlat: kernel execution failed");
}

void cuda_addmat(MatGPU &mat, const MatGPU &b) {
  _applyBinaryOp(mat, b, BinaryOp::Add());  
}

void cuda_submat(MatGPU &mat, const MatGPU &b) {
  _applyBinaryOp(mat, b, BinaryOp::Subtract());  
}

void cuda_multmat(MatGPU &mat, const MatGPU &b) {
  _applyBinaryOp(mat, b, BinaryOp::Multiply());  
}

void cuda_divmat(MatGPU &mat, const MatGPU &b) {
  _applyBinaryOp(mat, b, BinaryOp::Divide());  
}

void cuda_sigmder(MatGPU &mat, const MatGPU &b) {
  _applyBinaryOp(mat, b, BinaryOp::SigmDer());  
}

/* ------- Conditional operations ------- */

template<class CondOp, class Op>
__global__ void kEltwiseCondOpFlat(const float* a, const float* condmat, bool incase,
                                   float* const dest, int numElements, CondOp condOp, Op op) {
    const int idxX = blockIdx.x * ELTWISE_FLAT_THREADS_X + threadIdx.x;
    if (incase) {
      for (int x = idxX; x < numElements; x += gridDim.x * ELTWISE_FLAT_THREADS_X) {
          if (condOp(condmat[x])) {
            dest[x] = op(a[x]);
          }
      }
    } else {
      for (int x = idxX; x < numElements; x += gridDim.x * ELTWISE_FLAT_THREADS_X) {
          if (!condOp(condmat[x])) {
            dest[x] = op(a[x]);
          }
      }
    }
}

template <class CondOp, class Op> 
void _applyCondOp(MatGPU& mat, const MatGPU& condmat, bool incase, CondOp condOp, Op op) {
    
    if (mat.empty()) return;     
    mexAssert(mat.stride_ == 1 && condmat.stride_ == 1, "In _applyCondOp strides should be 1");    
    mexAssert(mat.order_ == condmat.order_, "In _applyCondOp orders should be the same");    
    mexAssert(mat.size1_ == condmat.size1_ && mat.size2_ == condmat.size2_,
      "In _applyCondOp the sizes of matrices do not correspond");
    int _numElements = (int) (mat.size1_ * mat.size2_);    
    cudaStream_t stream = MatGPU::_defaultStream;    
    
    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
    kEltwiseCondOpFlat<CondOp, Op><<<blocks, threads, 0, stream>>>
      (mat.data_, condmat.data_, incase, mat.data_, _numElements, condOp, op);    
    mexAssert(cudaGetLastError() == cudaSuccess, "kEltwiseCondOpFlat: kernel execution failed");
}

void cuda_condassign(MatGPU& mat, const MatGPU& condmat, bool incase, float threshold, float val) {
  _applyCondOp(mat, condmat, incase, UnaryOp::BiggerThanScalar(threshold), UnaryOp::Scalar(val));
}

void cuda_condadd(MatGPU& mat, const MatGPU& condmat, bool incase, float threshold, float val) {
  _applyCondOp(mat, condmat, incase, UnaryOp::BiggerThanScalar(threshold), UnaryOp::AddScalar(val));
}

void cuda_condmult(MatGPU& mat, const MatGPU& condmat, bool incase, float threshold, float val) {
  _applyCondOp(mat, condmat, incase, UnaryOp::BiggerThanScalar(threshold), UnaryOp::MultByScalar(val));
}

/* ------- Softmax derivatives ------- */

__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, int numCases, int numOut) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];        
        dE_dx_l[tidx] = v;        
    }
}

void computeSoftmaxGrad(const MatGPU& acts, const MatGPU& actsGrad, MatGPU& target) {

    int numCases = (int) acts.size1_;
    int numOut = (int) acts.size2_;

    mexAssert(acts.stride_ == 1 && actsGrad.stride_ == 1 && target.stride_ == 1, "csg2");
    mexAssert(acts.size1_ == actsGrad.size1_ && acts.size2_ == actsGrad.size2_, "csg1");
    mexAssert(acts.size1_ == target.size1_ && acts.size2_ == target.size2_, "csg3");
    
    cudaStream_t stream = MatGPU::_defaultStream;
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));    
    kSoftmaxGrad<<<blocks, threads, 0, stream>>>(actsGrad.data_, acts.data_, target.data_, numCases, numOut);
    mexAssert(cudaGetLastError() == cudaSuccess, "computeSoftmaxGrad: kernel execution failed");    
}

/* ------- Transposition ------- */

/*
 * dest here is assumed to be "not transposed" -- height and width correspond to it.
 */
template<class Op, bool checkBounds>
__global__ void kEltwiseUnaryOpTrans(const float* a, float* const dest,
                                     int height, int width, int strideA, int strideDest, Op op) {

    __shared__ float shmem[ELTWISE_THREADS_X][ELTWISE_THREADS_X + 1];

    for (int by = ELTWISE_THREADS_X * blockIdx.y; by < height; by += ELTWISE_THREADS_X * gridDim.y) {
        for (int bx = ELTWISE_THREADS_X * blockIdx.x; bx < width; bx += ELTWISE_THREADS_X * gridDim.x) {
            const int readX = by + threadIdx.x;
            const int readY = bx + threadIdx.y;
            for (int y = 0; y < ELTWISE_THREADS_X; y+= ELTWISE_THREADS_Y) {
                if (!checkBounds || (readX < height && readY + y < width)) {
                    shmem[threadIdx.x][threadIdx.y + y] = op(a[(readY + y) * strideA + readX]);
                }
            }
            __syncthreads();

            const int writeX = bx + threadIdx.x;
            const int writeY = by + threadIdx.y;
            for (int y = 0; y < ELTWISE_THREADS_X; y+= ELTWISE_THREADS_Y) {
                if(!checkBounds || (writeX < width && writeY + y < height)) {
                    dest[(writeY + y) * strideDest + writeX] = shmem[threadIdx.y + y][threadIdx.x];

                }
            }
            __syncthreads();
        }
    }
}

void cuda_trans(const MatGPU &mat, MatGPU &target) {

    mexAssert(mat.order_ == target.order_, "In cuda_trans orders should be the same");
    mexAssert(mat.size1_ == target.size2_ && mat.size2_ == target.size1_,
      "In cuda_trans sizes does not correspond to each other");    
      
    int width = (int) target.size1_;
    int height = (int) target.size2_;
    
    int mat_stride = (int) mat.size1_;
    int target_stride = (int) target.size1_;
    if (mat.order_ == true) {
      mat_stride = (int) mat.size2_;
      target_stride = (int) target.size2_;
    }

    cudaStream_t stream = MatGPU::_defaultStream;

    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
    bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
    if (checkBounds) {
        kEltwiseUnaryOpTrans<UnaryOp::Identity, true><<<blocks, threads, 0, stream>>>
          (mat.data_, target.data_, height, width, mat_stride, target_stride, UnaryOp::Identity());
    } else {
        kEltwiseUnaryOpTrans<UnaryOp::Identity, false><<<blocks, threads, 0, stream>>>
          (mat.data_, target.data_, height, width, mat_stride, target_stride, UnaryOp::Identity());
    }
    mexAssert(cudaGetLastError() == cudaSuccess, "kEltwiseUnaryOpTrans: kernel execution failed");    
}

/* ------- Matrix <-> Vector operations ------- */

/*
 * Matrix in ROW-MAJOR order!
 */
template <class Op>
__global__ void kRowVectorOp(const float* mat, const float* vec, float* const tgtMat, 
                             int width, int height, int matStride, int tgtStride, Op op) {
    __shared__ float shVec[ADD_VEC_THREADS_X];
    const int bx = ADD_VEC_THREADS_X * blockIdx.x;
    const int by = ADD_VEC_THREADS_Y * blockIdx.y;

    for (int x = bx; x < width; x += gridDim.x * ADD_VEC_THREADS_X) {
        __syncthreads();
        if (x + threadIdx.x < width && threadIdx.y == 0) {
            shVec[threadIdx.x] = vec[x + threadIdx.x];
        }
        __syncthreads();

        if (x + threadIdx.x < width) {
            for (int y = by + threadIdx.y; y < height; y += gridDim.y * ADD_VEC_THREADS_Y) {
                tgtMat[y * tgtStride + x + threadIdx.x] = op(mat[y * matStride + x + threadIdx.x], shVec[threadIdx.x]);
            }
        }
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
 
template <class Op>
__global__ void kColVectorOp(float* mat, float* vec, float* tgtMat,
                             int width, int height, int matStride, int tgtStride, Op op) {
    __shared__ float shVec[ADD_VEC_THREADS_Y];
    const int by = ADD_VEC_THREADS_Y * blockIdx.y;
    const int bx = ADD_VEC_THREADS_X * blockIdx.x;
    const int tidx = ADD_VEC_THREADS_X * threadIdx.y + threadIdx.x;
    
    mat += threadIdx.y * matStride;
    vec += tidx;
    tgtMat += threadIdx.y * tgtStride;

    for (int y = by; y < height; y += gridDim.y * ADD_VEC_THREADS_Y) {
        __syncthreads();
        if (y + tidx < height && tidx < ADD_VEC_THREADS_Y) {
            shVec[tidx] = vec[y];
        }
        __syncthreads();

        if (y + threadIdx.y < height) {
            for (int x = bx + threadIdx.x; x < width; x += gridDim.x * ADD_VEC_THREADS_X) {
                tgtMat[(y) * tgtStride + x] = op(mat[(y) * matStride + x], shVec[threadIdx.y]);
            }
        }
    }
}

template <class Op> 
void _applyBinaryV(MatGPU &mat, const MatGPU &vect, size_t dim, Op op) {  

  mexAssert(mat.data_ != vect.data_, "av1");
  mexAssert(mat.stride_ == 1 && vect.stride_ == 1, "av2");
  
  int width = (int) mat.size1_;
  int height = (int) mat.size2_;

  dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);  
  cudaStream_t stream = MatGPU::_defaultStream;
  
  if (dim == 1) {
    mexAssert(vect.size1_ == 1 && vect.size2_ == mat.size2_, "In '_applyBinaryV' the sizes don't correspond");
    dim3 blocks(std::min(512, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
    kColVectorOp<Op><<<blocks, threads, 0, stream>>>(mat.data_, vect.data_, mat.data_, width, height, width, width, op);
  }   
  else if (dim == 2) {
    /* actually not used, but let it be here just in case */
    mexAssert(vect.size1_ == mat.size1_ && vect.size2_ == 1, "In '_applyBinaryV' the sizes don't correspond");
    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
    kRowVectorOp<Op><<<blocks, threads, 0, stream>>>(mat.data_, vect.data_, mat.data_, width, height, width, width, op);    
  } else {
    mexAssert(false, "_applyBinaryV the dimension parameter must be either 1 or 2");
  }
  mexAssert(cudaGetLastError() == cudaSuccess, "_applyBinaryV: kernel execution failed");
}

void cuda_addvect(MatGPU &mat, const MatGPU &vect, size_t dim) {
  _applyBinaryV(mat, vect, dim, BinaryOp::Add());
}

void cuda_multvect(MatGPU &mat, const MatGPU &vect, size_t dim) {
  _applyBinaryV(mat, vect, dim, BinaryOp::Multiply());
}

/*
 * To be used when the rows are <= 64.
 *
 * TODO: try to reduce reg usage. i think this can be made faster too.
 */
//#define AGG_SHORT_ROWS_LOOPS_X  4
template <class Agg, class UnaryOp, class BinaryOp, int LOOPS_X, int THREADS_X>
__global__ void kAggShortRows(const float* mat, float* matSum, int width, int height, Agg agg, UnaryOp uop, BinaryOp bop) {
    const int shmemX = THREADS_X + 1;
    __shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];

    const int tidx = threadIdx.y * THREADS_X + threadIdx.x;
    const int ty = LOOPS_X == 1 ? tidx / width : threadIdx.y; // when loops==1, width is gonna be smaller than block x dim
    const int tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;
    float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
    matSum += blockRowIdx + tidx;
//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
    mat += width * blockRowIdx + MUL24(ty, width) + tx;
    float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];

    bool doAgg = tidx < AGG_SHORT_ROWS_THREADS_Y ;

    if (blockRowIdx < height) {
        #pragma unroll
        for (int y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
            doAgg &= tidx + y + blockRowIdx < height;
            const bool heightIdxOK = ty < AGG_SHORT_ROWS_THREADS_Y && ty + y + blockRowIdx < height;

            shmemWriteZeros[0] = agg.getBaseValue();
            __syncthreads();
            #pragma unroll
            for(int x = 0; x < LOOPS_X * THREADS_X; x+= THREADS_X) {
//                __syncthreads();
                if (heightIdxOK && x + tx < width) {
                    shmemWrite[0] = agg(uop(mat[x]), shmemWrite[0]);
                }
            }
            __syncthreads();
            if (doAgg) {
                /*
                 * I tried doing this final sum as a 4-step reduction, with 8 threads
                 * per warp participating. It was slightly slower.
                 */
                float accum = agg.getBaseValue();
                float* shmemRead = shmem + MUL24(tidx, shmemX);
                // this loops too much if the rows are really short :(
                #pragma unroll
                for (int i = 0; i < THREADS_X; i++) {
                    accum = agg(accum, shmemRead[0]);
                    shmemRead++;
                }
                matSum[0] = bop(matSum[0], accum);
                matSum += AGG_SHORT_ROWS_THREADS_Y;
            }
            __syncthreads();
            mat += width * AGG_SHORT_ROWS_THREADS_Y;
        }
    }
}

template <class Agg, class UnaryOp, class BinaryOp>
__global__ void kAggShortRows2(const float* mat, float* matSum, int width, int height, Agg agg, UnaryOp uop, BinaryOp bop) {
    const int shmemX = AGG_SHORT_ROWS_THREADS_X + 1;
    __shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];
    const int LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);
    const int tidx = threadIdx.y * AGG_SHORT_ROWS_THREADS_X + threadIdx.x;

    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;

    float* shmemWrite = shmem + MUL24(threadIdx.y, shmemX) + threadIdx.x;
    matSum += blockRowIdx + tidx;
//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
    mat += width * blockRowIdx + MUL24(threadIdx.y, width) + threadIdx.x;

    bool doAgg = tidx < AGG_SHORT_ROWS_THREADS_Y;
    if(blockRowIdx < height) {
        for (int y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
            doAgg &= tidx + y + blockRowIdx < height;
            const bool heightIdxOK = threadIdx.y + y + blockRowIdx < height;
            float accum = agg.getBaseValue();
            shmemWrite[0] = agg.getBaseValue();

            for(int x = 0; x < LOOPS_X * AGG_SHORT_ROWS_THREADS_X; x+= AGG_SHORT_ROWS_THREADS_X) {
//                __syncthreads();
                if (heightIdxOK && x + threadIdx.x < width) {
                    shmemWrite[0] = agg(uop(mat[x]), shmemWrite[0]);
                }
            }

            __syncthreads();
            if (doAgg) {
                float* shmemRead = shmem + MUL24(tidx, shmemX);

                #pragma unroll
                for (int i = 0; i < AGG_SHORT_ROWS_THREADS_X; i++) {
                    accum = agg(accum, shmemRead[0]);
                    shmemRead++;
                }

                matSum[0] = bop(matSum[0], accum);
                matSum += AGG_SHORT_ROWS_THREADS_Y;
            }
            __syncthreads();
            mat += width * AGG_SHORT_ROWS_THREADS_Y;
        }
    }
}

/*
 * Implements multiscan idea from http://www.moderngpu.com
 * Not really useful for pure reductions but neat nonetheless.
 */
template<class Agg, class UnaryOp, class BinaryOp>
__global__ void kAggRows_wholerow_nosync(const float* mat, float* matSum, int width, int height,
                                         Agg agg, UnaryOp uop, BinaryOp bop) {
    const int tidx = threadIdx.x;
    const int warpIdx = tidx / WARP_SIZE;
    const int lane = tidx % WARP_SIZE;
    
    __shared__ float accum[(WARP_SIZE + 1) * AWR_NUM_WARPS];
    __shared__ float finalAccum[AWR_NUM_WARPS];

    float* myAccum = &accum[warpIdx * (WARP_SIZE + 1) + lane];
    float* myFinalAccum = &finalAccum[tidx];
    //volatile float* vMyAccum = &accum[warpIdx * (WARP_SIZE + 1) + lane];
    matSum += blockIdx.y;
    mat += width * blockIdx.y;

    float rAccum = agg.getBaseValue(); // cache in register, a bit faster than shmem
    #pragma unroll 32
    for (int x = tidx; x < width; x += AWR_NUM_THREADS) {
        rAccum = agg(rAccum, uop(mat[x]));
    }
    myAccum[0] = rAccum;
    
    // Each warp does a reduction that doesn't require synchronizatoin
    #pragma unroll
    for (int i = 0; i < LOG_WARP_SIZE; i++) {
        const int d = 1 << i;
        myAccum[0] = agg(myAccum[0], shfl_down(myAccum[0], d));
    }
    __syncthreads();
    // The warps write their results
    if (tidx < AWR_NUM_WARPS) {
        //volatile float* vMyFinalAccum = &finalAccum[tidx];
        myFinalAccum[0] = accum[tidx * (WARP_SIZE + 1)];
        #pragma unroll
        for (int i = 0; i < AWR_LOG_NUM_WARPS; i++) {
            const int d = 1 << i;
            myFinalAccum[0] = agg(myFinalAccum[0], shfl_down(myFinalAccum[0], d));
        }
        if (tidx == 0) {
            matSum[0] = bop(matSum[0], myFinalAccum[0]);
            matSum += gridDim.y;
        }
    }
}

/*
 * This one gets coalesced reads but computes only a partial sum which
 * must either be summed again (recursively) or summed on the host.
 */
template<class Agg, class UnaryOp, class BinaryOp, int blockSize>
__global__ void kAggRows(const float* mat, float* matSum, int width, int height, int sumWidth, Agg agg, UnaryOp uop, BinaryOp bop) {
    const int idxX = blockIdx.x * blockSize*2 + threadIdx.x;

    __shared__ float accum[blockSize*2];

    matSum += blockIdx.y * sumWidth + blockIdx.x;
    /*
     * Here it's important to make sure that all threads in a block call __syncthreads,
     * so I have even the redundant threads (for which idxX >= width) enter this loop
     * just so that they may call __syncthreads at the appropriate times.
     */
    mat += width * blockIdx.y + idxX;

    accum[threadIdx.x] = agg.getBaseValue();
    accum[threadIdx.x + blockSize] = agg.getBaseValue();
    for (int idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
        if (idxX < width) {
            accum[threadIdx.x] = uop(mat[0]);
            if(idxX + blockSize < width)
                accum[threadIdx.x + blockSize] = uop(mat[blockSize]);
        }
        if (blockSize >= 512) {
            __syncthreads();
            if (threadIdx.x < 512)
                accum[threadIdx.x] = agg(accum[threadIdx.x], accum[threadIdx.x + 512]);
        }
        if (blockSize >= 256) {
            __syncthreads();
            if (threadIdx.x < 256)
                accum[threadIdx.x] = agg(accum[threadIdx.x],accum[threadIdx.x + 256]);
        }
        if (blockSize >= 128) {
            __syncthreads();
            if (threadIdx.x < 128)
                accum[threadIdx.x] = agg(accum[threadIdx.x],accum[threadIdx.x + 128]);
        }
        if (blockSize >= 64) {
            __syncthreads();
            if (threadIdx.x < 64)
                accum[threadIdx.x] = agg(accum[threadIdx.x],accum[threadIdx.x + 64]);
        }

        __syncthreads();
        volatile float* myAccum = &accum[threadIdx.x];
        if (threadIdx.x < 32) { // executed only by first warp
            myAccum[0] = agg(myAccum[0], myAccum[32]);
            myAccum[0] = agg(myAccum[0], myAccum[16]);
            myAccum[0] = agg(myAccum[0], myAccum[8]);
            myAccum[0] = agg(myAccum[0], myAccum[4]);
            myAccum[0] = agg(myAccum[0], myAccum[2]);
            myAccum[0] = agg(myAccum[0], myAccum[1]);
        }

        if (threadIdx.x == 0) {
            matSum[0] = bop(matSum[0], myAccum[0]);
            matSum += gridDim.y * sumWidth;
        }
        __syncthreads();
        mat += width * gridDim.y;
    }
}

/*
 * Bad when there are few columns.
 */
template <class Agg, class UnaryOp, class BinaryOp>
__global__ void kDumbAggCols(cudaTextureObject_t mat, float* const vec, int width, int height, Agg agg, UnaryOp uop, BinaryOp bop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        float mx = agg.getBaseValue();
        for (int j = 0; j < height; j++) {
            mx = agg(uop(tex1Dfetch<float>(mat, width * j + idx)), mx);
        }
        vec[idx] = bop(vec[idx], mx);
    }
}

/*
 * Better with few columns because it only computes a partial sum.
 */
template <class Agg, class UnaryOp>
__global__ void kAggCols(cudaTextureObject_t mat, float* const vec, int width, int height, int sumLength, Agg agg, UnaryOp op) {
    const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY = blockIdx.y * sumLength;
    if (idxX < width) {
        float mx = agg.getBaseValue();
        for (int j = idxY; j < min(height,idxY + sumLength); j++) {
            mx = agg(op(tex1Dfetch<float>(mat, j * width + idxX)), mx);
        }
        vec[blockIdx.y * width + idxX] = mx;
    }
}

/*
 * TODO: this is a mess, fix it. it works pretty fast but it's too ugly.
 * TODO: this function is _really_ bad for very long aggregations of few columns.
 */
template<class Agg, class UOp, class BOp>
void _aggregate(MatGPU &mat, MatGPU& target, Agg agg, UOp uop, BOp bop, int axis) {

    mexAssert(axis == 0 || axis == 1, "ag1");
    mexAssert(mat.stride_ == 1 && target.stride_ == 1, "ag2");
    mexAssert(mat.data_ != target.data_, "ag3");
    mexAssert(!mat.empty(), "ag4");
    
    int width = (int) mat.size1_;
    int height = (int) mat.size2_;

    cudaStream_t stream = MatGPU::_defaultStream;
    
    if (axis == 0 ) { //sum along size2_
        mexAssert(target.size1_ == mat.size1_ && target.size2_ == 1, "ag5");
        if ((height <= 2048 || width >= 4096)) {
            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            mexAssert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width, "ag6");
            mexAssert(numBlocks < NUM_BLOCKS_MAX, "ag7");
            kDumbAggCols<Agg, UOp, BOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(mat.getTextureObject(), target.data_, width, height, agg, uop, bop);
            mexAssert(cudaGetLastError() == cudaSuccess, "kDumbAggCols: kernel execution failed");            
        } else { // Specialize the case when we have very long columns and few of them
            const int sumLength = 128;
            MatGPU tmp(width, DIVUP(height, sumLength));
            int numBlocksX = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            int numBlocksY = DIVUP(height, sumLength);
            dim3 blocks(numBlocksX, numBlocksY);
            dim3 threads(NUM_SUM_COLS_THREADS_PER_BLOCK);
            kAggCols<Agg, UOp><<<blocks,threads, 0, stream>>>(mat.getTextureObject(), tmp.data_, width, height, sumLength, agg, uop);
            mexAssert(cudaGetLastError() == cudaSuccess, "kAggCols: kernel execution failed");                        

            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            kDumbAggCols<Agg, UOp, BOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(tmp.getTextureObject(), target.data_, width, height, agg, uop, bop);
            mexAssert(cudaGetLastError() == cudaSuccess, "kDumbAggCols: kernel execution failed");            
        }      
    } else { // sum along size1_
        mexAssert(target.size1_ == 1 && target.size2_ == mat.size2_, "ag8");
        if (width > 1) {
            if (height >= 16384) { // linear aggregation
                int numBlocksX = 1;
                int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 : width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
                int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                while (numBlocksY > NUM_BLOCKS_MAX) {
                    numBlocksY = DIVUP(numBlocksY, 2);
                    numBlocksX *= 2;
                }
                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                if(width <= 16) {
                    if(width <= 4) {
                        kAggShortRows<Agg, UOp, BOp, 1, 4><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                    } else if(width <= 8) {
                        kAggShortRows<Agg, UOp, BOp, 1, 8><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                    } else if(width <= 12) {
                        kAggShortRows<Agg, UOp, BOp, 1, 12><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                    } else {
                        kAggShortRows<Agg, UOp, BOp, 1, 16><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                    }
                } else if(width <= 32) {
                    kAggShortRows<Agg, UOp, BOp, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                } else if(width <= 48){
                    kAggShortRows<Agg, UOp, BOp, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                } else if(width <= 64){
                    kAggShortRows<Agg, UOp, BOp, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                } else {
                    kAggShortRows2<Agg, UOp, BOp><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                }
            } else {
                if (width >= 512) {
                    // NOTE: this is the only case which I bothered to try to optimize for Kepler
                    dim3 threads(AWR_NUM_THREADS);
                    dim3 blocks(1, height);
                    kAggRows_wholerow_nosync<<<blocks, threads, 0, stream>>>(mat.data_, target.data_, width, height, agg, uop, bop);
                } else {

                    int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    int numThreadsY = 1;
                    int numBlocksX = DIVUP(width, 2*numThreadsX);
                    int numBlocksY = std::min(height, NUM_BLOCKS_MAX);

                    dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                    mexAssert(numBlocksX <= NUM_BLOCKS_MAX, "ag9");
                    mexAssert(numBlocksY <= NUM_BLOCKS_MAX, "ag10");

                    if(width <= 64) {
                        kAggRows<Agg, UOp, BOp, 32><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, (int) target.size1_, agg, uop, bop);
                    } else if(width <= 128) {
                        kAggRows<Agg, UOp, BOp, 64><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, (int) target.size1_, agg, uop, bop);
                    } else if(width <= 256) {
                        kAggRows<Agg, UOp, BOp, 128><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, (int) target.size1_, agg, uop, bop);
                    } else if(width <= 512) {
                        kAggRows<Agg, UOp, BOp, 256><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, (int) target.size1_, agg, uop, bop);
                    } else {
                        kAggRows<Agg, UOp, BOp, 512><<<grid, threads, 0, stream>>>(mat.data_, target.data_, width, height, (int) target.size1_, agg, uop, bop);
                    }
                    mexAssert(cudaGetLastError() == cudaSuccess, "agg rows: kernel execution failed");
                }
            }
        } else {
            mexAssert(false, "fake aggregation, use assignment instead");
            //target.applyBinary(NVMatrixBinaryOps::CompositeSecond<UOp, BOp>(uop, bop), *this, target, stream);
        }
    }
}

void cuda_sumvect(MatGPU &mat, MatGPU &vect, size_t dim) {
  int axis = 2 - (int) dim;  
  _aggregate(mat, vect, Aggs::Sum(), UnaryOp::Identity(), BinaryOp::Second(), axis);  
}

void cuda_maxvect(MatGPU &mat, MatGPU &vect, size_t dim) {
  int axis = 2 - (int) dim;
  _aggregate(mat, vect, Aggs::Max(), UnaryOp::Identity(), BinaryOp::Second(), axis);  
}

/* ------------- CrossMapResponseNormLayer ------------- */

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 *
 * So each block does one pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y
 */
template<int B_Y, int B_X, int imgsPerThread, bool checkCaseBounds, bool blocked>
__global__ void kFCNorm(cudaTextureObject_t imgs, cudaTextureObject_t meanDiffs, float* target, const int imgSize,
                          const int numFilters, const int numImages, const int sizeF,
                          const float addScale, const float powScale, const float minDiv) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;

    
    const int imgIdx = blockImgIdx + threadIdx.x;
    const int imgOffset = ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    const int meanDiffsOffset = pxIdx * numImages + imgIdx;
//    imgs += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
//    meanDiffs += pxIdx * numImages + imgIdx;
    target += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    
    float prod[imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            prod[i] = 0;
        }
    }

    const int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF/2 + filterIdx;
    const int loopStartF = blocked ? startF : MAX(0, startF);
    const int loopEndF = MIN(numFilters, startF + sizeF);
 
    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                float val = tex1Dfetch<float>(meanDiffs, meanDiffsOffset + f * imgPixels * numImages + i * B_X);
                prod[i] += val * val;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            prod[i] = minDiv + addScale * prod[i];
            target[i * B_X] = tex1Dfetch<float>(imgs, imgOffset + i * B_X) * __powf(prod[i], -powScale);
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 *
 * So each block does one output pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y
 *
 * TODO: this is pretty wasteful of computation. a lot of threads basically compute the same products.
 */
template<int B_Y, int B_X, int imgsPerThread, bool add, bool checkCaseBounds, bool blocked>
//__launch_bounds__(128,16)
__global__ void kFRNormUndo2(cudaTextureObject_t outGrads, cudaTextureObject_t inputs, cudaTextureObject_t acts, float* target, const int imgSize, const int numFilters, const int numImages, const int sizeF, const float addScale, const float powScale, const float minDiv, const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;

    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;

    const int imgPixels = imgSize * imgSize;
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + threadIdx.x;

    const int inpOffset = pxIdx * numImages + imgIdx;
    const int outOffset = ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;

    target += outOffset;

    float prod[imgsPerThread];
    float denoms[imgsPerThread];

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        prod[i] = 0;
        denoms[i] = 0;
    }

    int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF + sizeF/2 + 1 + filterIdx;
    int loopStartF = blocked ? startF : MAX(0, startF);
    int loopEndF = MIN(numFilters, startF + sizeF);

    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                // If an input is zero, then we shuldn't divide by it.
                const float grad = tex1Dfetch<float>(outGrads, inpOffset + f * imgPixels * numImages + i * B_X);
                const float act = tex1Dfetch<float>(acts, inpOffset + f * imgPixels * numImages + i * B_X);
                const float inp = tex1Dfetch<float>(inputs, inpOffset + f * imgPixels * numImages + i * B_X) + (act == 0);
                prod[i] += grad * act * __powf(__fdividef(act, inp), 1.0f/powScale);
            }
        }
    }

    startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF/2 + filterIdx;
    loopStartF = blocked ? startF : MAX(0, startF);
    loopEndF = MIN(numFilters, startF + sizeF);

    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                float val = tex1Dfetch<float>(inputs, inpOffset + f * imgPixels * numImages + i * B_X);
                denoms[i] += val * val;
            }
        }
    }

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = tex1Dfetch<float>(inputs, outOffset + i * B_X);
                const float out = tex1Dfetch<float>(outGrads, outOffset + i * B_X);
                denoms[i] = addScale * denoms[i] + minDiv;
                prod[i] = (-2 * powScale * addScale * inp * prod[i] + out * __powf(denoms[i], -powScale));
                target[i * B_X] = prod[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = tex1Dfetch<float>(inputs, outOffset + i * B_X);
                const float out = tex1Dfetch<float>(outGrads, outOffset + i * B_X);
                denoms[i] = addScale * denoms[i] + minDiv;
                prod[i] = (-2 * powScale * addScale * inp * prod[i] + out * __powf(denoms[i], -powScale));
                target[i * B_X] = scaleTargets * target[i * B_X] + scaleOutputs * prod[i];
            }
        }
    }
}

/*
 * images:      (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)

 * Note: at present, I have no code to compute the meanDiffs. So it should be set
 * to be equal to images. In other words, this isn't really doing contrast normalization,
 * just response normalization.
 */
void _convContrastNormCrossMap(MatGPU& images, MatGPU& meanDiffs, MatGPU& target,
                              size_t imgSize1, size_t imgSize2, size_t normsize, float addScale, float powScale) {
                              
    bool blocked = false;
    float minDiv = 1.0f;

    int sizeF = (int) normsize;
    int imgSizeX = (int) imgSize1;    
    int imgSizeY = (int) imgSize2;
    mexAssert(imgSizeX == imgSizeY, "In cmrnorm layer the images should be squared");    
    int imgSize = imgSizeX;
    int imgPixels = imgSizeX * imgSizeY;
    
    int numImages = (int) images.size1_;
    mexAssert(images.size2_ % imgPixels == 0, "cnc1");    
    int numFilters = (int) images.size2_ / imgPixels;    
    
    mexAssert(meanDiffs.size1_ == images.size1_, "cnc2");
    mexAssert(meanDiffs.size2_ == images.size2_, "cnc3");
    
    mexAssert(target.size1_ == images.size1_, "cnc5");
    mexAssert(target.size2_ == images.size2_, "cnc6");
                             
    mexAssert(0 < sizeF && sizeF <= numFilters, "cnc4");
    mexAssert(numFilters % 16 == 0, "Number of outputmaps should be divisible by 16");
    
    cudaStream_t stream = MatGPU::_defaultStream;

    bool checkCaseBounds = numImages % 128 != 0;

    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / 4) * imgSize);    
    if (blocked) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, true, true>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, true, true><<<blocks, threads, 0, stream>>>(images.getTextureObject(), meanDiffs.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        } else {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, false, true>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, false, true><<<blocks, threads, 0, stream>>>(images.getTextureObject(), meanDiffs.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        }
    } else {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, true, false>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, true, false><<<blocks, threads, 0, stream>>>(images.getTextureObject(), meanDiffs.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        } else {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, false, false>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, false, false><<<blocks, threads, 0, stream>>>(images.getTextureObject(), meanDiffs.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        }
    }
    mexAssert(cudaGetLastError() == cudaSuccess, "convContrastNormCrossMap: kernel execution failed");    
}

/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 *
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
void _convResponseNormCrossMapUndo(MatGPU& images, MatGPU& acts, MatGPU& outGrads, MatGPU& target, 
                                  size_t imgSize1, size_t imgSize2, size_t normsize, float addScale, float powScale) {
                         
    bool blocked = false;
    float scaleTargets = 0.0f;
    float scaleOutput = 1.0f;
    float minDiv = 1.0f;

    int sizeF = (int) normsize;
    int imgSizeX = (int) imgSize1;    
    int imgSizeY = (int) imgSize2;
    mexAssert(imgSizeX == imgSizeY, "In cmrnorm layer the images should be squared");    
    int imgSize = imgSizeX;
    int imgPixels = imgSizeX * imgSizeY;
    
    int numImages = (int) outGrads.size1_;
    mexAssert(outGrads.size2_ % imgPixels == 0, "cnc_undo1");    
    int numFilters = (int) outGrads.size2_ / imgPixels;    
    
    mexAssert(images.size1_ == outGrads.size1_, "cnc_undo2");
    mexAssert(images.size2_ == outGrads.size2_, "cnc_undo3");
    
    mexAssert(acts.size1_ == outGrads.size1_, "cnc_undo4");
    mexAssert(acts.size2_ == outGrads.size2_, "cnc_undo5");
        
    mexAssert(target.size1_ == outGrads.size1_, "cnc_undo6");
    mexAssert(target.size2_ == outGrads.size2_, "cnc_undo7");    
    
    mexAssert(0 < sizeF && sizeF <= numFilters, "cnc_undo8");
    mexAssert(numFilters % 16 == 0, "Number of outputmaps should be divisible by 16");
    
    cudaStream_t stream = MatGPU::_defaultStream;

    dim3 threads2 = dim3(32, 4);
    dim3 blocks2 = dim3(DIVUP(numImages,32*4) * imgSize, (numFilters / 4) * imgSize);

    bool checkCaseBounds = (numImages % 128) != 0;
    if (blocked) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, true, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, true, true><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, false, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, false, true><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, true, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, true, true><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, false, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, false, true><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            }
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, true, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, true, false><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, false, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, false, false><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, true, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, true, false><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, false, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, false, false><<<blocks2, threads2, 0, stream>>>(outGrads.getTextureObject(), images.getTextureObject(), acts.getTextureObject(), target.data_, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, scaleTargets, scaleOutput);
            }
        }
    }
    mexAssert(cudaGetLastError() == cudaSuccess, "convResponseNormCrossMapUndo: kernel execution failed");
}

/* ------- Sergey Demyanov ------- */

/* ------- Image jittering ------- */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread>
__global__ void kTransform(float* imgs, float* targets, int imgSizeX, int imgSizeY, int outputsX, int outputsY,
                           int numFilters, int numImages, float *shift_mat, float *scale_mat, float *mirror_mat, float *angle_mat, float defval) {
                           
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
  const int outputIdxX = blockIdx.x / numImgBlocks;
  const int outputIdxY = blockIdx.y / numFilterBlocks;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
  const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
  if (myFilterIdx >= numFilters) {
      return;
  }
  
  const int outputIdx = outputIdxY * outputsX + outputIdxX;
  const int numOutputs = outputsX * outputsY;
  const int imgPixels = imgSizeX * imgSizeY;
  
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += myFilterIdx * imgPixels * numImages + imgIdx;
  targets += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
    
  const float m1 = (float) imgSizeX / 2 - 0.5;
  const float m2 = (float) imgSizeY / 2 - 0.5;  
  const float n1 = (float) outputsX / 2 - 0.5;
  const float n2 = (float) outputsY / 2 - 0.5;
    
  // #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    const int curImgIdx = imgIdx + i * B_X;
    if (curImgIdx < numImages) {
      const float angcos = (float) cos(angle_mat[curImgIdx]);
      const float angsin = (float) sin(angle_mat[curImgIdx]);    
      const float xi1 = (outputIdxX - n1) * scale_mat[curImgIdx]; // scale[0];
      const float xi2 = (outputIdxY - n2) * scale_mat[curImgIdx + numImages]; //scale[1];
      float x1 = xi1 * angcos - xi2 * angsin + m1 + shift_mat[curImgIdx]; //shift[0];
      float x2 = xi1 * angsin + xi2 * angcos + m2 + shift_mat[curImgIdx + numImages]; //shift[1];
      if (mirror_mat[curImgIdx] > 0.5) x1 = imgSizeX - 1 - x1;
      if (mirror_mat[curImgIdx + numImages] > 0.5) x2 = imgSizeY - 1 - x2;
      /* hack for MNIST starts */
      /*
      x1 = MAX(x1, 0);
      x1 = MIN(x1, imgSizeX - 1);
      x2 = MAX(x2, 0);
      x2 = MIN(x2, imgSizeY - 1);
      */
      /* hack for MNIST ends */
      if (0 <= x1 && x1 <= imgSizeX - 1 &&
          0 <= x2 && x2 <= imgSizeY - 1) {
        const int xu1 = (int) x1;
        const int xu2 = (int) x2;      
        const int xp1 = MIN(xu1 + 1, imgSizeX - 1);
        const int xp2 = MIN(xu2 + 1, imgSizeY - 1);
        const int imgPx11 = xu2 * imgSizeX + xu1;
        const int imgPx21 = xu2 * imgSizeX + xp1;
        const int imgPx12 = xp2 * imgSizeX + xu1;
        const int imgPx22 = xp2 * imgSizeX + xp1;
        for (int f = 0; f < filtersPerThread; f++) {
          const int imgInd11 = (f * imgPixels + imgPx11) * numImages + i * B_X;
          const int imgInd21 = (f * imgPixels + imgPx21) * numImages + i * B_X;
          const int imgInd12 = (f * imgPixels + imgPx12) * numImages + i * B_X;
          const int imgInd22 = (f * imgPixels + imgPx22) * numImages + i * B_X;
          const float vl = (x1 - (float) xu1) * imgs[imgInd21] + ((float) xu1 + 1 - x1) * imgs[imgInd11];
          const float vh = (x1 - (float) xu1) * imgs[imgInd22] + ((float) xu1 + 1 - x1) * imgs[imgInd12];
          prod[f][i] = (x2 - (float) xu2) * vh + ((float) xu2 + 1 - x2) * vl;
        }
      } else {
        /* if (xf1 < 0) {
        } else if (imgSizeX <= xf1 + 1) {
        }
        if (xf2 < 0) {
        } else if (imgSizeY <= xf2 + 1) {
        } */
        for (int f = 0; f < filtersPerThread; f++) {
          prod[f][i] = defval;
        }
      }      
    }
  }
  
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
      if (imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
              targets[f * numOutputs * numImages + i * B_X] = prod[f][i];
          }
      }
  }
}


void _transformActs(const MatGPU &images, MatGPU &target, 
                    size_t imgSize1, size_t imgSize2,
                    size_t targSize1, size_t targSize2,
                    const MatGPU &shift_mat, const MatGPU &scale_mat, 
                    const MatGPU &mirror_mat, const MatGPU &angle_mat, float defval) {
                    
  int imgSizeX = (int) imgSize1;    
  int imgSizeY = (int) imgSize2;    
  int imgPixels = imgSizeX * imgSizeY;
  int outputsX = (int) targSize1;
  int outputsY = (int) targSize2;
  int targPixels = outputsX * outputsY;
  
  int numImages = (int) images.size1_;
  mexAssert(images.size2_ % imgPixels == 0, "ta2");
  int numFilters = (int) images.size2_ / imgPixels;
  
  mexAssert(target.size1_ == numImages, "ta1");
  mexAssert(target.size2_ == targPixels * numFilters, "ta3");
  
  cudaStream_t stream = MatGPU::_defaultStream;
  
  int filtersPerThread = 1;
  int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
  dim3 threads(32, 4);
  dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsY);
  
  if (imgsPerThread == 4) {
    kTransform<4, 32, 4, 1><<<blocks, threads, 0, stream>>>
      (images.data_, target.data_, imgSizeX, imgSizeY, outputsX, outputsY, numFilters, numImages, 
       shift_mat.data_, scale_mat.data_, mirror_mat.data_, angle_mat.data_, defval);       
  } else if (imgsPerThread == 2) {
    kTransform<4, 32, 2, 1><<<blocks, threads, 0, stream>>>
      (images.data_, target.data_, imgSizeX, imgSizeY, outputsX, outputsY, numFilters, numImages, 
       shift_mat.data_, scale_mat.data_, mirror_mat.data_, angle_mat.data_, defval);
  } else {
    kTransform<4, 32, 1, 1><<<blocks, threads, 0, stream>>>
      (images.data_, target.data_, imgSizeX, imgSizeY, outputsX, outputsY, numFilters, numImages, 
       shift_mat.data_, scale_mat.data_, mirror_mat.data_, angle_mat.data_, defval);
  }
  mexAssert(cudaGetLastError() == cudaSuccess, "_transformActs: kernel execution failed");                        
}

/* ------- Total matrix aggregation ------- */


/*
 * Sergey Demyanov
 * This kernel is faster than the one in the code of Alex Krizhevsky, so I'll leave it here
 */

__global__
void _totalSum(const float *a, float* const b, size_t n) {
  __shared__ float sdata[DP_BLOCKSIZE];  
  size_t tid = threadIdx.x;
  sdata[tid] = 0;
  size_t gridSize = blockDim.x * gridDim.x;
  size_t i = blockDim.x * blockIdx.x + tid;  
  if (i >= n) return;
  while (i < n) {
    sdata[tid] += a[i];
    i += gridSize;
  }
  __syncthreads();
  // do reduction in shared mem
  size_t ns = blockDim.x; // number of elements in the shared array
  if (ns > n - blockDim.x * blockIdx.x ) {
    ns = n - blockDim.x * blockIdx.x;
  }
  size_t s = blockDim.x; // sum stride
  while (s >= ns) s >>= 1; 
  while (s > 32) {
    if (tid < s && tid + s < ns) sdata[tid] += sdata[tid + s];
    __syncthreads();
     s >>= 1;
  } 
  // for s <= WARP_SIZE no synchronization is needed
  if (tid < 32) {    
    if (tid + 32 < ns) sdata[tid] += sdata[tid + 32];
    if (tid + 16 < ns) sdata[tid] += sdata[tid + 16];
    if (tid + 8 < ns) sdata[tid] += sdata[tid + 8];
    if (tid + 4 < ns) sdata[tid] += sdata[tid + 4];
    if (tid + 2 < ns) sdata[tid] += sdata[tid + 2];
    if (tid + 1 < ns) sdata[tid] += sdata[tid + 1];
  }  
  // write result for this block to global mem
  if (tid == 0) b[blockIdx.x] = sdata[0];
}

float cuda_sum(const MatGPU &mat) {

  mexAssert(!mat.empty(), "In cuda_sum mat is empty");
  mexAssert(mat.stride_ == 1, "In cuda_sum stride_ should be 1");
  
  cudaStream_t stream = MatGPU::_defaultStream;
  
  size_t numElements = mat.size1_ * mat.size2_;
  size_t blocks_number = MIN(DIVUP(numElements, ELTWISE_FLAT_THREADS_X), ELTWISE_FLAT_THREADS_X);
  //MatGPU::_sum_buf1.resize(ELTWISE_FLAT_THREADS_X, 1);
  //MatGPU::_sum_buf2.resize(1, 1);
  MatGPU partsums, totalsum;
  MatGPU::swapWithBuffer(partsums, ELTWISE_FLAT_THREADS_X);
  partsums.resize(ELTWISE_FLAT_THREADS_X, 1);
  MatGPU::swapWithBuffer(totalsum, 1);
  totalsum.resize(1, 1);
  _totalSum<<<(unsigned int) blocks_number, ELTWISE_FLAT_THREADS_X, 0, stream>>>
    (mat.data_, partsums.data_, numElements);
  _totalSum<<<1, ELTWISE_FLAT_THREADS_X, 0, stream>>>
    (partsums.data_, totalsum.data_, blocks_number);
  MatCPU cpusum(1, 1);
  DeviceToHost(totalsum, cpusum);
  MatGPU::swapWithBuffer(partsums, ELTWISE_FLAT_THREADS_X);
  MatGPU::swapWithBuffer(totalsum, 1);
  return cpusum(0, 0);
}

