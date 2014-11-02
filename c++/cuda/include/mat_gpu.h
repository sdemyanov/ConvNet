/*
Copyright (C) 2014 Sergey Demyanov. 
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

#ifndef _MAT_GPU_H_
#define _MAT_GPU_H_

#include "mat_cpu.h"
#include <cublas_v2.h>
#include <curand.h>
//#include <mutex>

#ifndef _Pragma // Windows
  #define _Pragma(x) __pragma(x)
#endif

class MatGPU : public MatCPU {

private:
  // static
  ftype *textdata_;
  cudaTextureObject_t texture_;

  static curandGenerator_t _randGen;
  static cudaStream_t _defaultStream;
  static cublasHandle_t _cublasHandle;  
  static int getDeviceID();
  
  static cudaEvent_t _start, _stop;  
  
  static MatGPU _subset_buf, _sum_buf;
  static MatGPU _softmax_buf1, _softmax_buf2;
  
public:
  // static
  static void CudaInit();
  static void InitRand(size_t seed);  
  static void CudaReset();  
  
  static void StartCudaTimer();
  static void MeasureCudaTime(std::string msg);
  
  // data access
  // private
  ftype operator () (size_t i, size_t j) const;
  size_t getNumDataBytes() const;
  cudaTextureObject_t getTextureObject();
  // public
  ftype operator () (size_t ind) const;
  MatGPU operator () (size_t ind);  
  
  // memory functions  
  MatGPU();  
  MatGPU(const std::vector<size_t> &newsize);
  MatGPU(size_t size1, size_t size2);
  MatGPU(const MatGPU &a);
  MatGPU(MatGPU &&a);
  ~MatGPU();  
  MatGPU& init();
  MatGPU& operator = (const MatGPU &a);
  MatGPU& resize(size_t size1, size_t size2);  
  MatGPU& reshape(size_t size1, size_t size2); 
  MatGPU& clear();
  friend void Swap(MatGPU &a, MatGPU &b);
  
  // data functions  
  MatGPU& assign(ftype val);
  MatGPU& rand();
  MatGPU& operator += (const MatGPU &a);
  MatGPU& operator -= (const MatGPU &a);
  MatGPU& operator *= (const MatGPU &a);
  MatGPU& operator /= (const MatGPU &a);
  MatGPU& operator += (ftype a);
  MatGPU& operator -= (ftype a);
  MatGPU& operator *= (ftype a);
  MatGPU& operator /= (ftype a);  
  MatGPU& Sign();
  MatGPU& Sqrt();  
  MatGPU& SoftMax();
  MatGPU& SoftDer(const MatGPU& a);
  MatGPU& Sigmoid();
  MatGPU& SigmDer(const MatGPU& a);
  MatGPU& CondAssign(const MatGPU &condMatGPU, bool incase, ftype threshold, ftype a);
  MatGPU& CondAdd(const MatGPU &condMatGPU, bool incase, ftype threshold, ftype a);
  MatGPU& CondMult(const MatGPU &condMatGPU, bool incase, ftype threshold, ftype a);
  MatGPU& AddVect(const MatGPU &vect, size_t dim);
  MatGPU& AddVect(const MatGPU &vect, const MatGPU &mask, size_t dim);
  MatGPU& MultVect(const MatGPU &vect, size_t dim);
  MatGPU& Normalize(ftype norm);
  MatGPU& Validate();
  
  // CPU <-> GPU functions
  MatGPU& operator = (const MatCPU &a); // HostToDevice
  friend void DeviceToHost(const MatGPU &b, MatCPU &a);  
  friend void SubSet(MatCPU &a, MatGPU &b, size_t offset, bool dir);
  
  // friend functions  
  
  friend void Sum(MatGPU &a, MatGPU &vect, size_t dim);
  friend void Mean(MatGPU &a, MatGPU &vect, size_t dim);
  
  friend void Trans(const MatGPU &a, MatGPU &b);
  friend void InitMaps(const MatGPU &a, const std::vector<size_t> &mapsize,
                       std::vector< std::vector<MatGPU> > &matrices);    
  
  // layer transformation functions
  friend void Prod(const MatGPU &a, bool a_tr, const MatGPU &b, bool b_tr, MatGPU &c);
  friend void Prod(const MatGPU &a, bool a_tr, const MatGPU &b, bool b_tr, const MatGPU &mask, MatGPU &c);

  // filter functions  
  friend void FilterActs(MatGPU& images, MatGPU& filters, MatGPU& targets,
                         const std::vector<size_t> &prev_mapsize, size_t padding);
  friend void ImgActs(MatGPU& hidActs, MatGPU& filters, MatGPU& targets,
                      const std::vector<size_t> &prev_mapsize, size_t padding);                      
  friend void WeightActs(MatGPU& images, MatGPU& hidActs, MatGPU& targets,
                         const std::vector<size_t> &prev_mapsize, size_t padding, 
                         size_t filtersize, size_t sum_width, MatGPU& weights_der);
  
  // scaling functions  
  friend void AvgPooling(MatGPU& images, MatGPU& targets,
                         const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride);
  friend void MaxPooling(MatGPU& images, MatGPU& targets,
                         const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride);  
  friend void AvgPoolingUndo(MatGPU& avgGrads, MatGPU& targets,
                             const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride);
  friend void MaxPoolingUndo(MatGPU& images, MatGPU& maxActs, MatGPU& maxGrads, MatGPU& imgGrads,
                             const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride, bool dir);  
  
  // const functions
  ftype sum() const;  
                        
private:  

  // cuda_util.cu
  template <class Op> 
  friend void _applyUnaryOp(MatGPU &mat, Op op);
  template <class Op> 
  friend void _applyBinaryOp(MatGPU& mat, const MatGPU& b, Op op);
  template <class CondOp, class Op> 
  friend void _applyCondOp(MatGPU& mat, const MatGPU& condmat, bool incase, CondOp condOp, Op op);
  template <class Op>
  friend void _applyBinaryV(MatGPU &mat, const MatGPU &vect, const MatGPU &mask, size_t dim, Op op);
  template<class Agg, class UnaryOp, class BinaryOp>
  friend void _aggregate(MatGPU &mat, MatGPU& target, Agg agg, UnaryOp uop, BinaryOp bop, int axis);
  friend void computeSoftmaxGrad(const MatGPU& acts, const MatGPU& actsGrad, MatGPU& target);
  friend void cuda_trans(const MatGPU &mat, MatGPU &target);
  friend float cuda_sum(const MatGPU &mat);
  
  // filter_acts.cu
  friend void _filterActs(MatGPU& images, MatGPU& filters, MatGPU& targets,
                          size_t imgSize1, size_t imgSize2, size_t padding);						  
  // img_acts.cu
  friend void _imgActs(MatGPU& hidActs, MatGPU& filters, MatGPU& targets,
                       size_t imgSize1, size_t imgSize2, size_t padding);                
  // weight_acts.cu
  friend void _weightActs(MatGPU& images, MatGPU& hidActs, MatGPU& targets,
                         size_t imgSize1, size_t imgSize2, size_t padding, 
                         size_t chunks_num, size_t sum_width);  
  // scaled_acts.cu  
  template<class Pooler>
  friend void _convLocalPool(MatGPU& images, MatGPU& targets,
                             size_t imgSize1, size_t imgSize2, size_t scale, size_t stride, Pooler pooler);
  friend void _convLocalAvgPool(MatGPU& images, MatGPU& targets,
                                size_t imgSize1, size_t imgSize2, size_t scale, size_t stride);
  friend void _convLocalMaxPool(MatGPU& images, MatGPU& targets,
                                size_t imgSize1, size_t imgSize2, size_t scale, size_t stride);    
  friend void _convLocalMaxUndoDer(MatGPU& images, MatGPU&  maxActs, MatGPU& imgGrads, MatGPU& targets,
                                   size_t imgSize1, size_t imgSize2, size_t scale, size_t stride);
  //restored_acts.cu
  friend void _convLocalAvgUndo(MatGPU& avgGrads, MatGPU& targets,
                                size_t imgSize1, size_t imgSize2, size_t scale, size_t stride);
  friend void _convLocalMaxUndo(MatGPU& images, MatGPU& maxActs, MatGPU& maxGrads, MatGPU& targets, 
                                size_t imgSize1, size_t imgSize2, size_t scale, size_t stride);
  
};

#endif