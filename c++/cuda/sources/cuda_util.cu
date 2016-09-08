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

#define THREADS_PER_BLOCK                 128
#define THREADS_PER_DIM_1                  16
#define THREADS_PER_DIM_2                  16
#define IMAGES_PER_THREAD                  32
#define CHANNELS_PER_THREAD                32
#define DP_BLOCKSIZE                      512


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

// ------- Unary operations ------- //

template<class Op>
__global__ void kEltwiseUnaryOpFlat(const float* a, float* const dest, int num_elements, Op op) {
    const int idxX = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
        dest[x] = op(a[x]);
    }
}

template <class Op>
void _applyUnaryOp(MatGPU &mat, Op op) {

    if (mat.empty()) return;
    int num_elements = mat.size();
    cudaStream_t stream = MatGPU::_defaultStream;

    dim3 threads = dim3(THREADS_PER_BLOCK);
    dim3 blocks = dim3(DIVUP(num_elements, THREADS_PER_BLOCK));
    kEltwiseUnaryOpFlat<Op><<<blocks, threads, 0, stream>>>(mat.data_, mat.data_, num_elements, op);
    CUDA_CALL(cudaGetLastError());
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

/* ------- Create Identity Matrix ------- */

__global__ void kInitIdentityMatrix(float* a, int size, int num_elements) {
    const int idxX = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
        if (x % size == x / size) {
            a[x] = 1;
        } else {
            a[x] = 0;
        }
    }
}

void cuda_ident(MatGPU &mat) {

  if (mat.empty()) return;
  mexAssertMsg(mat.size1_ == mat.size2_,
    "In 'cuda_ident' the matrix must be squared");

  int width = mat.size1_;
  int num_elements = mat.size();
  cudaStream_t stream = MatGPU::_defaultStream;

  dim3 threads = dim3(THREADS_PER_BLOCK);
  dim3 blocks = dim3(DIVUP(num_elements, THREADS_PER_BLOCK));
  kInitIdentityMatrix<<<blocks, threads, 0, stream>>>(mat.data_, width, num_elements);
  CUDA_CALL(cudaGetLastError());

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
__global__ void kEltwiseBinaryOpFlat(const float* a, const float* b, float* const dest, int num_elements, Op op) {
    const int idxX = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
        dest[x] = op(a[x], b[x]);
    }
}

template <class Op>
void _applyBinaryOp(MatGPU& mat, const MatGPU& b, Op op) {

    if (mat.empty()) return;
    mexAssertMsg(mat.order_ == b.order_, "In _applyBinaryOp orders should be the same");
    mexAssertMsg(mat.size1_ == b.size1_ && mat.size2_ == b.size2_,
      "In _applyBinaryOp the sizes of matrices do not correspond");
    int num_elements = mat.size();
    cudaStream_t stream = MatGPU::_defaultStream;

    dim3 threads = dim3(THREADS_PER_BLOCK);
    dim3 blocks = dim3(DIVUP(num_elements, THREADS_PER_BLOCK));
    kEltwiseBinaryOpFlat<Op><<<blocks, threads, 0, stream>>>(mat.data_, b.data_, mat.data_, num_elements, op);
    CUDA_CALL(cudaGetLastError());
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
                                   float* const dest, int num_elements, CondOp condOp, Op op) {
    const int idxX = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (incase) {
      for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
          if (condOp(condmat[x])) {
            dest[x] = op(a[x]);
          }
      }
    } else {
      for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
          if (!condOp(condmat[x])) {
            dest[x] = op(a[x]);
          }
      }
    }
}

template <class CondOp, class Op>
void _applyCondOp(MatGPU& mat, const MatGPU& condmat, bool incase, CondOp condOp, Op op) {

    if (mat.empty()) return;
    mexAssertMsg(mat.order_ == condmat.order_, "In _applyCondOp orders should be the same");
    mexAssertMsg(mat.size1_ == condmat.size1_ && mat.size2_ == condmat.size2_,
      "In _applyCondOp the sizes of matrices do not correspond");
    int num_elements = mat.size();
    cudaStream_t stream = MatGPU::_defaultStream;

    dim3 threads = dim3(THREADS_PER_BLOCK);
    dim3 blocks = dim3(DIVUP(num_elements, THREADS_PER_BLOCK));
    kEltwiseCondOpFlat<CondOp, Op><<<blocks, threads, 0, stream>>>
      (mat.data_, condmat.data_, incase, mat.data_, num_elements, condOp, op);
    CUDA_CALL(cudaGetLastError());
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

__global__
void _totalSum(const float *a, float* const b, int n) {
  __shared__ float sdata[DP_BLOCKSIZE];
  int tid = threadIdx.x;
  sdata[tid] = 0;
  int grid_size = blockDim.x * gridDim.x;
  int i = blockDim.x * blockIdx.x + tid;
  if (i >= n) return;
  while (i < n) {
    sdata[tid] += a[i];
    i += grid_size;
  }
  __syncthreads();
  // do reduction in shared mem
  int ns = blockDim.x; // number of elements in the shared array
  if (ns > n - blockDim.x * blockIdx.x ) {
    ns = n - blockDim.x * blockIdx.x;
  }
  int s = blockDim.x; // sum stride
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

  mexAssertMsg(!mat.empty(), "In cuda_sum mat is empty");

  cudaStream_t stream = MatGPU::_defaultStream;

  int threads = THREADS_PER_BLOCK;
  int num_elements = mat.size();
  int blocks = DIVUP(num_elements, threads);
  //MatGPU::_sum_buf1.resize(threads, 1);
  //MatGPU::_sum_buf2.resize(1, 1);
  MatGPU partsums, totalsum(1, 1);
  partsums.GetFromWorkspace(1, threads);
  _totalSum<<<dim3(blocks), dim3(threads), 0, stream>>>
    (mat.data_, partsums.data_, num_elements);
  _totalSum<<<1, dim3(threads), 0, stream>>>
    (partsums.data_, totalsum.data_, blocks);
  MatCPU cpusum(1, 1);
  DeviceToHost(totalsum, cpusum);
  return cpusum(0, 0);
}

// ------- Sergey Demyanov ------- //

// ------- Image jittering ------- //

__global__
void kTransform(float* imgs, float* targets,
  int imgSize1, int imgSize2, int trgSize1, int trgSize2,
  int numChannels, int numImages, int channelBlocks,
  float *shift_mat, float *scale_mat, float *mirror_mat, float *angle_mat,
  float defval, bool dir) {

  const int targetIdx1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int targetIdx2 = blockIdx.y * blockDim.y + threadIdx.y;
  const int channelBlockIdx = blockIdx.z % channelBlocks;
  const int imageBlockIdx = blockIdx.z / channelBlocks;
  // HW layout
  const int targetIdx = targetIdx1 * trgSize2 + targetIdx2;

  if (targetIdx1 >= trgSize1 || targetIdx2 >= trgSize2) {
    return;
  }

  const int imgPixels = imgSize1 * imgSize2;
  const int trgPixels = trgSize1 * trgSize2;

  float buffer[IMAGES_PER_THREAD][CHANNELS_PER_THREAD];

  const float imgHalf1 = (float) imgSize1 / 2 - 0.5;
  const float imgHalf2 = (float) imgSize2 / 2 - 0.5;
  const float trgHalf1 = (float) trgSize1 / 2 - 0.5;
  const float trgHalf2 = (float) trgSize2 / 2 - 0.5;

  // #pragma unroll
  const int first_im = imageBlockIdx * IMAGES_PER_THREAD;
  const int last_im = MIN(first_im + IMAGES_PER_THREAD, numImages);
  const int first_ch = channelBlockIdx * CHANNELS_PER_THREAD;
  const int last_ch = MIN(first_ch + CHANNELS_PER_THREAD, numChannels);
  for (int i = first_im; i < last_im; ++i) {
    // indices for (n, 2) param arrays
    const int i0 = 2*i, i1 = 2*i+1;
    float x1, x2;
    if (dir == true) { // forward
      const float xi1 = (targetIdx1 - trgHalf1) * scale_mat[i0]; // scale[0];
      const float xi2 = (targetIdx2 - trgHalf2) * scale_mat[i1]; //scale[1];
      const float angcos = (float) cos(angle_mat[i]);
      const float angsin = (float) sin(angle_mat[i]);
      x1 = xi1 * angcos - xi2 * angsin + imgHalf1 + shift_mat[i0]; //shift[0];
      x2 = xi1 * angsin + xi2 * angcos + imgHalf2 + shift_mat[i1]; //shift[1];
      if (mirror_mat[i0] > 0.5) x1 = imgSize1 - 1 - x1;
      if (mirror_mat[i1] > 0.5) x2 = imgSize2 - 1 - x2;
    } else {
      int mt1 = targetIdx1, mt2 = targetIdx2;
      if (mirror_mat[i0] > 0.5) mt1 = trgSize1 - 1 - mt1;
      if (mirror_mat[i1] > 0.5) mt2 = trgSize2 - 1 - mt2;
      const float xi1 = mt1 - trgHalf1 - shift_mat[i0];
      const float xi2 = mt2 - trgHalf2 - shift_mat[i1];
      const float angcos = (float) cos(-angle_mat[i]);
      const float angsin = (float) sin(-angle_mat[i]);
      x1 = (xi1 * angcos - xi2 * angsin) / scale_mat[i0] + imgHalf1; //shift[0];
      x2 = (xi1 * angsin + xi2 * angcos) / scale_mat[i1] + imgHalf2; //shift[1];
    }
    if (0 <= x1 && x1 <= imgSize1 - 1 &&
        0 <= x2 && x2 <= imgSize2 - 1) {
      const int xu1 = (int) (x1 + 0.5);
      const int xu2 = (int) (x2 + 0.5);
      const int xp1 = MIN(xu1 + 1, imgSize1 - 1);
      const int xp2 = MIN(xu2 + 1, imgSize2 - 1);
      // HW layout
      const int imgPx11 = xu1 * imgSize2 + xu2;
      const int imgPx21 = xu1 * imgSize2 + xp2;
      const int imgPx12 = xp1 * imgSize2 + xu2;
      const int imgPx22 = xp1 * imgSize2 + xp2;
      for (int c = first_ch; c < last_ch; ++c) {
        // NCHW or NCWH layout
        const int cur_shift = (i * numChannels + c) * imgPixels;
        const int imgInd11 = cur_shift + imgPx11;
        const int imgInd21 = cur_shift + imgPx21;
        const int imgInd12 = cur_shift + imgPx12;
        const int imgInd22 = cur_shift + imgPx22;
        const float vl = (x1 - (float) xu1) * imgs[imgInd21] + ((float) xu1 + 1 - x1) * imgs[imgInd11];
        const float vh = (x1 - (float) xu1) * imgs[imgInd22] + ((float) xu1 + 1 - x1) * imgs[imgInd12];
        buffer[i-first_im][c-first_ch] = (x2 - (float) xu2) * vh + ((float) xu2 + 1 - x2) * vl;
      }
    } else {
      for (int c = 0; c < numChannels; ++c) {
        buffer[i-first_im][c-first_ch] = defval;
      }
    }
  }

  // shift to the current pixel
  targets += targetIdx;
  #pragma unroll
  for (int i = first_im; i < last_im; ++i) {
    #pragma unroll
    for (int c = first_ch; c < last_ch; ++c) {
      targets[(i * numChannels + c) * trgPixels] = buffer[i-first_im][c-first_ch];
    }
  }
}

void _affineTransform(const MatGPU &images, MatGPU &targets,
                    int imgSize1, int imgSize2,
                    int trgSize1, int trgSize2,
                    const MatGPU &shift_mat, const MatGPU &scale_mat,
                    const MatGPU &mirror_mat, const MatGPU &angle_mat,
                    float defval, bool dir) {

  // dir = true -> forward
  // dir = false -> backward

  int numImages = images.size1_;

  int imgPixels = imgSize1 * imgSize2;
  mexAssertMsg(images.size2_ % imgPixels == 0, "ta2");
  int numChannels = images.size2_ / imgPixels;

  mexAssertMsg(targets.size1_ == numImages, "ta1");
  mexAssertMsg(targets.size2_ == trgSize1 * trgSize2 * numChannels, "ta3");

  cudaStream_t stream = MatGPU::_defaultStream;

  dim3 threads(THREADS_PER_DIM_1, THREADS_PER_DIM_2);

  int imageBlocks = DIVUP(numImages, IMAGES_PER_THREAD);
  int channelBlocks = DIVUP(numChannels, CHANNELS_PER_THREAD);
  dim3 blocks(DIVUP(trgSize1, THREADS_PER_DIM_1),
              DIVUP(trgSize2, THREADS_PER_DIM_2),
              imageBlocks * channelBlocks);

  kTransform<<<blocks, threads, 0, stream>>>
    (images.data_, targets.data_, imgSize1, imgSize2, trgSize1, trgSize2,
     numChannels, numImages, channelBlocks,
     shift_mat.data_, scale_mat.data_, mirror_mat.data_, angle_mat.data_,
     defval, dir);

  CUDA_CALL(cudaGetLastError());
}

/*
 * Sergey Demyanov
 * Function for the 3rd step max pooling propagation
 * derivs here are the derivatives for images and have the same size as activs,
 * pool_derivs have the same size as pool_activs, but get values from derivs
 * propagating values from the same positions as on the first pass
 */
__global__
void kMaxPoolThirdPass(float* activs, float* pool_activs, float* derivs, float* pool_derivs,
                       int imgSize1, int imgSize2, int trgSize1, int trgSize2,
                       int numChannels, int numImages, int channelBlocks,
                       int scale1, int scale2, int stride1, int stride2) {

  const int targetIdx1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int targetIdx2 = blockIdx.y * blockDim.y + threadIdx.y;
  const int channelBlockIdx = blockIdx.z % channelBlocks;
  const int imageBlockIdx = blockIdx.z / channelBlocks;
  // HW layout
  const int targetIdx = targetIdx1 * trgSize2 + targetIdx2;
  // shift to current pixel, so later we do only image and channel shifts
  pool_activs += targetIdx;
  pool_derivs += targetIdx;
  // coordinate of the window upper left corner
  const int imageIdx = targetIdx1 * stride1 * imgSize2 + targetIdx2 * stride2;
  activs += imageIdx;
  derivs += imageIdx;

  if (targetIdx1 >= trgSize1 || targetIdx2 >= trgSize2) {
    return;
  }

  const int imgPixels = imgSize1 * imgSize2;
  const int trgPixels = trgSize1 * trgSize2;

  float buffer[IMAGES_PER_THREAD][CHANNELS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < IMAGES_PER_THREAD; ++i) {
    #pragma unroll
    for (int c = 0; c < CHANNELS_PER_THREAD; ++c) {
      buffer[i][c] = 0;
    }
  }

  const int first_im = imageBlockIdx * IMAGES_PER_THREAD;
  const int last_im = MIN(first_im + IMAGES_PER_THREAD, numImages);
  const int first_ch = channelBlockIdx * CHANNELS_PER_THREAD;
  const int last_ch = MIN(first_ch + CHANNELS_PER_THREAD, numChannels);
  for (int i = first_im; i < last_im; ++i) {
    for (int c = first_ch; c < last_ch; ++c) {
      // NCHW or NCWH layout. Shift to current channel
      const int channel_shift = i * numChannels + c;
      // shift to the pixel has been done in the beginning
      const float pooled_val = pool_activs[channel_shift * trgPixels];
      // guaranteed to be inside, no outside of the image checks
      bool found = false;
      for (int y = 0; y < scale1; ++y) {
        for (int x = 0; x < scale2; ++x) {
          int img_shift = channel_shift * imgPixels + y * imgSize2 + x;
          if (activs[img_shift] == pooled_val) {
            buffer[i-first_im][c-first_ch] += derivs[img_shift];
            found = true;
            break;
          }
        }
        if (found) break;
      }
    }
  }
  // shifted to pixel in the beginning
  #pragma unroll
  for (int i = first_im; i < last_im; ++i) {
    #pragma unroll
    for (int c = first_ch; c < last_ch; ++c) {
      pool_derivs[(i * numChannels + c) * trgPixels] = buffer[i-first_im][c-first_ch];
    }
  }
}



void _maxPoolThirdPass(const MatGPU& activs, const MatGPU&  pool_activs,
                       const MatGPU& derivs, MatGPU& pool_derivs,
                       int imgSize1, int imgSize2, int trgSize1, int trgSize2,
                       Pair scale, Pair padding, Pair stride) {

  // implement non-zero padding if necessary
  mexAssert(padding[0] == 0 && padding[1] == 0);

  // it is assumed that all size asserts have been done before
  int numImages = activs.size1_;
  int imgPixels = imgSize1 * imgSize2;
  mexAssertMsg(activs.size2_ % imgPixels == 0, "mpf1");
  int numChannels = activs.size2_ / imgPixels;

  cudaStream_t stream = MatGPU::_defaultStream;

  dim3 threads(THREADS_PER_DIM_1, THREADS_PER_DIM_2);

  int imageBlocks = DIVUP(numImages, IMAGES_PER_THREAD);
  int channelBlocks = DIVUP(numChannels, CHANNELS_PER_THREAD);
  dim3 blocks(DIVUP(trgSize1, THREADS_PER_DIM_1),
              DIVUP(trgSize2, THREADS_PER_DIM_2),
              imageBlocks * channelBlocks);

  kMaxPoolThirdPass<<<blocks, threads, 0, stream>>>(
    activs.data_, pool_activs.data_,
    derivs.data_, pool_derivs.data_,
    imgSize1, imgSize2, trgSize1, trgSize2,
    numChannels, numImages, channelBlocks,
    scale[0], scale[1], stride[0], stride[1]
  );

  CUDA_CALL(cudaGetLastError());
}


/*
// ------- Color variation ------- //

template<class Op>
__global__ void kRepMatOp(const float* a, const float* b, float* const dest, int size1, int size2,
                                     int bsize1, int bsize2, bool order, int dim, bool inner, Op op) {
    const int idxX = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const int num_elements = size1 * size2;

    for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
      int inda1, inda2, indb1, indb2, bx;
      if (order == false) { // Matlab order
        inda1 = x % size1;
        inda2 = x / size1;
      } else {
        inda1 = x / size2;
        inda2 = x % size2;
      }
      if (dim == 1) {
        indb2 = inda2;
        if (inner) {
          indb1 = inda1 / (size1 / bsize1);
        } else {
          indb1 = inda1 % bsize1;
        }
      } else if (dim == 2) {
        indb1 = inda1;
        if (inner) {
          indb2 = inda2 / (size2 / bsize2);
        } else {
          indb2 = inda2 % bsize2;
        }
      }
      if (order == false) { // Matlab order
        bx = indb2 * bsize1 + indb1;
      } else {
        bx = indb1 * bsize2 + indb2;
      }
      dest[x] = op(a[x], b[bx]);
    }
}

template <class Op>
void _applyRepMatOp(MatGPU& mat, const MatGPU& b, int dim, bool inner, Op op) {

  if (mat.empty() || b.empty()) return;
  mexAssertMsg(mat.order_ == b.order_, "In _applyRepMatOp orders should be the same");

  int num_elements = mat.size();
  cudaStream_t stream = MatGPU::_defaultStream;

  dim3 threads = dim3(THREADS_PER_BLOCK);
  dim3 blocks = dim3(std::min(128, DIVUP(num_elements, THREADS_PER_BLOCK)));


  if (dim == 1) {
    mexAssertMsg(mat.size2_ == b.size2_ && mat.size1_ % b.size1_ == 0,
    "In _applyRepMatOp the sizes of matrices do not correspond");
  } else if (dim == 2) {
    mexAssertMsg(mat.size1_ == b.size1_ && mat.size2_ % b.size2_ == 0,
    "In _applyRepMatOp the sizes of matrices do not correspond");
  } else {
    mexAssertMsg(false, "_applyRepMatOp the dimension parameter must be either 1 or 2");
  }
  kRepMatOp<Op><<<blocks, threads, 0, stream>>>(mat.data_, b.data_, mat.data_,
  mat.size1_, mat.size2_, b.size1_, b.size2_, mat.order_, dim, inner, op);

  mexAssertMsg(cudaGetLastError() == cudaSuccess, "_applyRepMatOp: kernel execution failed");
}

void _varyColors(MatGPU &images, const MatGPU &add_mat) {
  _applyRepMatOp(images, add_mat, 2, true, BinaryOp::Add());
}
*/
