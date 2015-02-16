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
 
#include "mat_gpu.h"
#include <math.h>

class AvgPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() const {
        return 0;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a / regionSize;
    }
};

class MaxPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fmaxf(a, b);
    }
    __device__ inline float getBaseValue() const {
        return -2e38; 
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

class MaxAbsPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fabsf(a) > fabsf(b) ? a : b;
    }
    __device__ inline float getBaseValue() const {
        return 0.0f;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * targets:      (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */
 
 template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool(float* imgs, float* targets, const int imgSizeX, const int imgSizeY, 
                           const int numFilters, const int numImages, const int subsX, const int startX,
                           const int strideX, const int outputsX, const int outputsY, Agg agg) {
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
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    targets += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSizeY, startImgPxY + subsX);
    const int loopEndX = MIN(imgSizeX, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSizeX + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
            }
        }
    }
}
 
/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does a 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * targets:      (numFilters, numOutputs, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 * 
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool2(float* imgs, float* targets, const int imgSizeX, const int imgSizeY,
                            const int numFilters,  const int numImages, const int subsX, 
                            const int startX, const int outputsX, const int outputsY, Agg agg) {
    __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
    const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
//    const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
    const int numOutputs = outputsX * outputsY;
    const int imgPixels = imgSizeX * imgSizeY;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int myX = threadIdx.y % 4;
    const int myY = threadIdx.y / 4;
    
    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;
    
    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;
    
    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSizeY, endImgPxY + 3);
    const int loopEndX = MIN(imgSizeX, endImgPxX + 3);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    targets += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++) {
        const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
        for (int x = loopStartX; x < loopEndX; x++) {
            // Load a pixel
            const int px = y * imgSizeX + x;
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Is this pixel in my region?
            if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] = agg(prod[f][i], shImgs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
                ++regionSize;
            }
            __syncthreads();

        }
    }
    if (myOutputIdxY < outputsY && myOutputIdxX < outputsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
                }
            }
        }
    }
}


/*
 * imgs:        (numFilters, imgPixels, numImages)
 * targets:      (numFilters, outputs, numImages)
 */
template<class Pooler>
void _convLocalPool(MatGPU& images, MatGPU& targets,
                    size_t imgSize1, size_t imgSize2, size_t scale, size_t stride, Pooler pooler) {
                    
    int subsX = (int) scale;
    int strideX = (int) stride;
    int imgSizeX = (int) imgSize1;    
    int imgSizeY = (int) imgSize2;    
    int imgPixels = imgSizeX * imgSizeY;
    int startX = 0;
    int outputsX = (int) DIVUP(imgSizeX, stride);    
    int outputsY = (int) DIVUP(imgSizeY, stride);
    int outputs = outputsX * outputsY;
    
    mexAssert(images.stride_ == 1 && targets.stride_ == 1,
            "In _convLocalPool one of strides is not 1"); 
                       
    int numImages = (int) images.size1_;
    
    mexAssert(images.size2_ % imgPixels == 0, "cp1");
    int numFilters = (int) images.size2_ / imgPixels;
    
    mexAssert(targets.size1_ == numImages, "cp2");
    mexAssert(targets.size2_ == outputs * numFilters, "cp3");    
    
    cudaStream_t stream = MatGPU::_defaultStream;

    if (strideX == 1 && subsX >= 6) {
        // NOTE: this part has not been optimized for Kepler
        int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
        int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        mexAssert((imgsPerThread * bx) % 32 == 0, "cp4");
        mexAssert(numFilters % filtersPerThread == 0, "cp5");
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(outputsX, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(outputsY, 4) * numFilters / filtersPerThread);
        if (imgsPerThread == 8) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            }
        } else if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, outputsX, outputsY, pooler);
                }
            }
        }
    } else {
        int filtersPerThread = numFilters % 16 == 0 ? 4 : 1;
        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        dim3 threads(32, 4);
        dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsY);
        if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 4, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 4, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 4, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 4, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                }
            }
        } else if (imgsPerThread == 2) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 4, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 4, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 4, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 4, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                }
            }
        } else {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 4, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 4, true><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 4, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 4, false><<<blocks, threads, 0, stream>>>(images.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, pooler);
                }
            }
        }
    }
    mexAssert(cudaGetLastError() == cudaSuccess, "convLocalPool: kernel execution failed");
}

void _convLocalAvgPool(MatGPU& images, MatGPU& targets,
                       size_t imgSize1, size_t imgSize2, size_t scale, size_t stride) {
  _convLocalPool(images, targets, imgSize1, imgSize2, scale, stride, AvgPooler());  
}

void _convLocalMaxPool(MatGPU& images, MatGPU& targets,
                       size_t imgSize1, size_t imgSize2, size_t scale, size_t stride) {
  _convLocalPool(images, targets, imgSize1, imgSize2, scale, stride, MaxPooler());  
}



/* 
 * Sergey Demyanov
 * Function for the 3rd step max pooling propagation
 * imgGrads here are the derivatives for images and have the same size,
 * targets have the same size as maxActs, but get values from imgGrads
 */

 template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalMaxUndoDer(float* imgs, float* maxActs, float* imgGrads, float* targets, 
                                 const int imgSizeX, const int imgSizeY, const int numFilters, const int numImages, 
                                 const int subsX, const int startX, const int strideX, const int outputsX, const int outputsY) {
                               
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
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    imgGrads += myFilterIdx * imgPixels * numImages + imgIdx;
    maxActs += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    targets += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSizeY, startImgPxY + subsX);
    const int loopEndX = MIN(imgSizeX, startImgPxX + subsX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSizeX + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const float img = imgs[(f * imgPixels + imgPx) * numImages + i * B_X];
                        const float ig = imgGrads[(f * imgPixels + imgPx) * numImages + i * B_X];
                        const float ma = maxActs[f * numOutputs * numImages + i * B_X];                        
                        prod[f][i] += (img == ma) * ig;                        
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                targets[f * numOutputs * numImages + i * B_X] = prod[f][i];
            }
        }
    }
}

void _convLocalMaxUndoDer(MatGPU& images, MatGPU&  maxActs, MatGPU& imgGrads, MatGPU& targets,
                          size_t imgSize1, size_t imgSize2, size_t scale, size_t stride) {
                    
    int subsX = (int) scale;
    int strideX = (int) stride;
    int imgSizeX = (int) imgSize1;    
    int imgSizeY = (int) imgSize2;    
    int imgPixels = imgSizeX * imgSizeY;
    int startX = 0;
    int outputsY = (int) DIVUP(imgSizeY, stride);
    int outputsX = (int) DIVUP(imgSizeX, stride);
    int outputs = outputsX * outputsY;
    
    mexAssert(images.stride_ == 1 && maxActs.stride_ == 1 && 
              imgGrads.stride_ == 1 && targets.stride_ == 1,
            "In _convLocalMaxUndoDer one of strides is not 1");            
                       
    int numImages = (int) maxActs.size1_;
    mexAssert(maxActs.size2_ % outputs == 0, "mud1");
    int numFilters = (int) maxActs.size2_ / outputs;    
    
    mexAssert(images.size1_ == numImages, "mud2");    
    mexAssert(images.size2_ == imgPixels * numFilters, "mud3"); 
    mexAssert(images.size1_ == imgGrads.size1_ &&
              images.size2_ == imgGrads.size2_, "mud4");
    mexAssert(maxActs.size1_ == targets.size1_ &&
              maxActs.size2_ == targets.size2_, "mud5");
              
    mexAssert(numFilters % 16 == 0, "Number of outputmaps should be divisible by 16");
    mexAssert(strideX <= subsX, "mud7");
    
    cudaStream_t stream = MatGPU::_defaultStream;

    int filtersPerThread = numFilters % 16 == 0 ? 4 : 1;
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsY);
    if (imgsPerThread == 4) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 4, 1, true><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            } else {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 4, 1, false><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 4, 4, true>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 4, 4, true><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            } else {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 4, 4, false>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 4, 4, false><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            }
        }
    } else if (imgsPerThread == 2) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 2, 1, true><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            } else {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 2, 1, false><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 2, 4, true>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 2, 4, true><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            } else {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 2, 4, false>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 2, 4, false><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            }
        }
    } else {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 1, 1, true><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            } else {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 1, 1, false><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 1, 4, true>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 1, 4, true><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            } else {
                cudaFuncSetCacheConfig(kLocalMaxUndoDer<4, 32, 1, 4, false>, cudaFuncCachePreferL1);
                kLocalMaxUndoDer<4, 32, 1, 4, false><<<blocks, threads, 0, stream>>>(images.data_, maxActs.data_, imgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY);
            }
        }
    }    
    mexAssert(cudaGetLastError() == cudaSuccess, "convLocalPool: kernel execution failed");
}

