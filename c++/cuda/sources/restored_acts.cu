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

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * targets:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalAvgUndo(float* avgGrads, float* targets, const int imgSizeX, const int imgSizeY, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX, const int outputsY, const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSizeX + blockPxX;
    const int numOutputs = outputsX * outputsY;
    const int imgPixels = imgSizeX * imgSizeY;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsY, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    avgGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    targets += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX && 
        blockPxY >= startX && blockPxY < startX + strideX * (outputsY-1) + subsX) {

        for (int my = startOutputY; my < endOutputY; my++) {
            const float regionStartY = fmaxf(0, startX + my * strideX);
            const float regionEndY = fminf(imgSizeY, startX + my * strideX + subsX);
            const float regionSizeY = regionEndY - regionStartY;
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                const float regionStartX = fmaxf(0, startX + mx * strideX);
                const float regionEndX = fminf(imgSizeX, startX + mx * strideX + subsX);
                const float regionSizeX = regionEndX - regionStartX;
                // It's important to do the division here, because pushing division into the below
                // loops makes the code 4x slower.
                const float regionSizeInv = 1.0f / (regionSizeX * regionSizeY);
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += avgGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X] * regionSizeInv;
                        }
                    }
                }
            }
        }
    }

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * maxActs:    (numFilters, numOutputs, numImages)
 * targets:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 * 
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalMaxUndo(float* imgs, float* maxGrads, float* maxActs, float* targets, 
                              const int imgSizeX, const int imgSizeY, const int numFilters, const int numImages, 
                              const int subsX, const int startX, const int strideX, const int outputsX,
                              const int outputsY, const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSizeX + blockPxX;
    const int numOutputs = outputsX * outputsY;
    const int imgPixels = imgSizeX * imgSizeY;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsY, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    maxActs += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    targets += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }    
    if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX && 
        blockPxY >= startX && blockPxY < startX + strideX * (outputsY-1) + subsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
                }
            }
        }
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            const float ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const float img = shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i];

                            prod[f][i] += (img == ma) * mg;
                        }
                    }
                }
            }
        }
    }
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    } 
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void _convLocalAvgUndo(MatGPU& avgGrads, MatGPU& targets,
                      size_t imgSize1, size_t imgSize2, size_t scale, size_t stride) {
                      
    int subsX = (int) scale;
    int strideX = (int) stride;
    int imgSizeX = (int) imgSize1;    
    int imgSizeY = (int) imgSize2;    
    int imgPixels = imgSizeX * imgSizeY;
    int startX = 0;
    int outputsX = DIVUP(imgSizeX, strideX);    
    int outputsY = DIVUP(imgSizeY, strideX);
    int outputs = outputsX * outputsY;
    ftype scaleTargets = 0;
    ftype scaleOutput = 1;
                      
    mexAssert(avgGrads.stride_ == 1 && targets.stride_ == 1,
            "In convLocalAvgUndo one of strides is not 1");            
                       
    int numImages = (int) avgGrads.size1_;
    mexAssert(avgGrads.size2_ % outputs == 0, "au1");
    int numFilters = (int) avgGrads.size2_ / outputs;    
    
    mexAssert(targets.size1_ == numImages, "au2");    
    mexAssert(targets.size2_ == imgPixels * numFilters, "au3");    
                      
    mexAssert(numFilters % 16 == 0, "Number of outputmaps should be divisible by 16");
    mexAssert(strideX <= subsX, "au5");
    
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSizeX, (numFilters / (4 * 4)) * imgSizeY);
    cudaStream_t stream = MatGPU::_defaultStream;
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, true><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, true><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, false><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, false><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, true><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, true><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, false><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, false><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        }
    } else {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, true><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, true><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, false><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, false><<<blocks, threads, 0, stream>>>(avgGrads.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        }
    }
    mexAssert(cudaGetLastError() == cudaSuccess, "convLocalAvgUndo: kernel execution failed");
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages) 
 * targets:      (numFilters, imgPixels, numImages)
 */
void _convLocalMaxUndo(MatGPU& images, MatGPU& maxActs, MatGPU& maxGrads, MatGPU& targets,                          size_t imgSize1, size_t imgSize2, size_t scale, size_t stride) {

    int subsX = (int) scale;
    int strideX = (int) stride;
    int imgSizeX = (int) imgSize1;
    int imgSizeY = (int) imgSize2;    
    int imgPixels = imgSizeX * imgSizeY;    
    int startX = 0;
    int outputsY = DIVUP(imgSizeY, strideX);
    int outputsX = DIVUP(imgSizeX, strideX);
    int outputs = outputsX * outputsY;
    ftype scaleTargets = 0;
    ftype scaleOutput = 1;
                      
    mexAssert(images.stride_ == 1 && maxActs.stride_ == 1 && 
              maxGrads.stride_ == 1 && targets.stride_ == 1,
            "In _convLocalMaxUndo one of strides is not 1");            
                       
    int numImages = (int) maxActs.size1_;
    mexAssert(maxActs.size2_ % outputs == 0, "mu1");
    int numFilters = (int) maxActs.size2_ / outputs;    
    
    mexAssert(targets.size1_ == numImages, "mu2");    
    mexAssert(targets.size2_ == imgPixels * numFilters, "mu3"); 
    mexAssert(images.size1_ == targets.size1_ &&
              images.size2_ == targets.size2_, "mu4");
    mexAssert(maxActs.size1_ == maxGrads.size1_ &&
              maxActs.size2_ == maxGrads.size2_, "mu5");
              
    mexAssert(numFilters % 16 == 0, "Number of outputmaps should be divisible by 16");
    mexAssert(strideX <= subsX, "mu7");
    
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSizeX, (numFilters / (4 * 2)) * imgSizeY);
    cudaStream_t stream = MatGPU::_defaultStream;
    if (imgsPerThread == 4) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 4, 2, false, true><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 4, 2, true, true><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 4, 2, false, false><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 4, 2, true, false><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 2, 2, false, true><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 2, 2, true, true><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 2, 2, false, false><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 2, 2, true, false><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        }
    } else {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 1, 2, false, true><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 1, 2, true, true><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 1, 2, false, false><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 1, 2, true, false><<<blocks, threads, 0, stream>>>(images.data_, maxGrads.data_, maxActs.data_, targets.data_, imgSizeX, imgSizeY, numFilters, numImages, subsX, startX, strideX, outputsX, outputsY, scaleTargets, scaleOutput);
            }
        }
    }
    mexAssert(cudaGetLastError() == cudaSuccess, "convLocalMaxUndo: kernel execution failed");
}

