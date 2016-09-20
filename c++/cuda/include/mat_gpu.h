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

#ifndef _MAT_GPU_H_
#define _MAT_GPU_H_

#include "mat_cpu.h"
#include "cuda_print.h"

#include <map>

#ifndef _Pragma // Windows
#define _Pragma(x) __pragma(x)
#endif

class MatGPU : public MatCPU {

private:
  // static
  cudnnTensorDescriptor_t tensor_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  void SetTensorDesc(const Dim &dims);
  void SetFilterDesc(const Dim &dims);
  cudnnTensorDescriptor_t GetTensorDesc();
  cudnnFilterDescriptor_t GetFilterDesc();
  void ClearTensorDesc();
  void ClearFilterDesc();

  static cudaStream_t _defaultStream;
  static curandGenerator_t _randGen;
  static cublasHandle_t _cublasHandle;
  static cudnnHandle_t _cudnnHandle;
  static int getDeviceID();
  static size_t _cudnnMemoryLimit;
  static MatGPU _workspace;

  static cudaEvent_t _start, _stop;

public:

  // static
  static void InitCuda(int gpu);
  static void InitRand(int seed);
  static void CudaReset();
  static void SetMemoryLimit(size_t memory);

  static void StartCudaTimer();
  static void MeasureCudaTime(std::string msg);

  // data access
  // private
  ftype operator () (size_t i, size_t j) const;
  size_t BytesNum() const;
  //cudaTextureObject_t getTextureObject();
  // public

  // memory functions
  MatGPU();
  MatGPU(size_t size1, size_t size2);
  MatGPU(const MatGPU &b);
  MatGPU(MatGPU &&b);
  ~MatGPU();
  MatGPU& init();
  MatGPU& operator = (const MatGPU &b);
  MatGPU& resize(size_t size1, size_t size2);
  MatGPU& resize_tensor(Dim dims);
  MatGPU& resize_filter(Dim dims);
  MatGPU& reshape(size_t size1, size_t size2);
  MatGPU& reshape_tensor(Dim dims);
  MatGPU& reshape_filter(Dim dims);
  MatGPU& attach(const MatGPU &b);
  MatGPU& attach(const MatGPU &b, size_t offset, size_t size1, size_t size2, bool order);
  MatGPU& attach(ftype *ptr, size_t size1, size_t size2);
  MatGPU& attach(ftype *ptr, size_t size1, size_t size2, bool order);
  MatGPU& clear();

  MatGPU& GetFromWorkspace(size_t size1, size_t size2);
  friend void Swap(MatGPU &a, MatGPU &b);

  // data functions
  MatGPU& ident();
  MatGPU& assign(ftype val);
  MatGPU& rand();
  MatGPU& randnorm();
  MatGPU& linear(ftype ca, ftype cb, const MatGPU &b, bool b_tr);
  MatGPU& operator += (const MatGPU &b);
  MatGPU& operator -= (const MatGPU &b);
  MatGPU& operator *= (const MatGPU &b);
  MatGPU& operator /= (const MatGPU &b);
  MatGPU& operator += (ftype c);
  MatGPU& operator -= (ftype c);
  MatGPU& operator *= (ftype c);
  MatGPU& operator /= (ftype c);
  MatGPU& Sign();
  MatGPU& Sqrt();
  MatGPU& Log();
  MatGPU& Exp();
  MatGPU& SoftMax();
  MatGPU& SoftDer(MatGPU& b);
  MatGPU& Sigmoid();
  MatGPU& SigmDer(const MatGPU& b);
  MatGPU& CondAssign(const MatGPU &condMatGPU, bool incase, ftype threshold, ftype a);
  MatGPU& CondAdd(const MatGPU &condMatGPU, bool incase, ftype threshold, ftype a);
  MatGPU& CondMult(const MatGPU &condMatGPU, bool incase, ftype threshold, ftype a);
  MatGPU& AddVect(MatGPU &vect, int dim);
  MatGPU& MultVect(const MatGPU &vect, int dim);
  MatGPU& Reorder(bool order);
  MatGPU& ReorderMaps(bool cur_order, bool order);
  MatGPU& Validate();

  // const functions
  Dim tensor_shape() const;
  Dim filter_shape() const;
  std::vector< std::vector<MatGPU> > InitMaps() const;
  ftype sum() const;

  // CPU <-> GPU functions
  MatGPU& operator = (const MatCPU &a); // HostToDevice
  friend void DeviceToHost(const MatGPU &b, MatCPU &a);
  friend void SubSet(MatCPU &a, MatGPU &b, size_t offset, bool dir);

  // friend functions

  friend void Sum(MatGPU &a, MatGPU &vect, int dim);
  friend void Mean(MatGPU &a, MatGPU &vect, int dim);
  //friend void Max(MatGPU &a, MatGPU &vect, int dim);

  friend void Trans(const MatGPU &a, MatGPU &b);


  // layer transformation functions
  friend void Prod(const MatGPU &a, bool a_tr, const MatGPU &b, bool b_tr, MatGPU &c);

  friend void AffineTransform(const MatGPU &images, MatGPU &targets,
                            const MatGPU &shift_mat, const MatGPU &scale_mat,
                            const MatGPU &mirror_mat, const MatGPU &angle_mat,
                            ftype defval, bool dir);
  /*
  friend void VaryColors(MatGPU &images, const std::vector<int> &mapsize,
                         const MatGPU &eigenvectors, ftype noise_std);

  */

  // CUDNN functions

  MatGPU& AddTensor(MatGPU &tensor);
  friend void ConvolutionForward(
    MatGPU& activs, MatGPU& filters, MatGPU& targets,
    const cudnnConvolutionDescriptor_t &conv_desc
  );
  friend void ConvolutionBackwardData(
    MatGPU& derivs, MatGPU& filters, MatGPU& targets,
    const cudnnConvolutionDescriptor_t &conv_desc
  );
  friend void ConvolutionBackwardFilter(
    MatGPU& activs, MatGPU& derivs, MatGPU& targets,
    const cudnnConvolutionDescriptor_t &conv_desc
  );
  friend void ConvolutionBackwardBias(
    MatGPU& derivs, MatGPU &targets
  );
  friend void Pooling(
    MatGPU& images, MatGPU& targets,
    cudnnPoolingDescriptor_t pool_desc
  );
  friend void PoolingUndo(
    MatGPU& activs, MatGPU& pool_activs,
    MatGPU& pool_derivs, MatGPU& targets,
    cudnnPoolingDescriptor_t pool_desc, bool dir
  );


private:
  // cuda_util.cu
  friend float cuda_sum(const MatGPU &mat);
  friend void cuda_ident(MatGPU &mat);
  template <class Op>
  friend void _applyUnaryOp(MatGPU &mat, Op op);
  template <class Op>
  friend void _applyBinaryOp(MatGPU& mat, const MatGPU& b, Op op);
  template <class CondOp, class Op>
  friend void _applyCondOp(MatGPU& mat, const MatGPU& condmat, bool incase, CondOp condOp, Op op);
  friend void _affineTransform(const MatGPU &images, MatGPU &targets,
                             int imgSize1, int imgSize2,
                             int targSize1, int targSize2,
                             const MatGPU &shift_mat, const MatGPU &scale_mat,
                             const MatGPU &mirror_mat, const MatGPU &angle_mat,
                             float defval, bool dir);

  friend void _maxPoolThirdPass(const MatGPU& activs, const MatGPU& pool_activs,
                                const MatGPU& derivs, MatGPU& pool_derivs,
                                int imgSize1, int imgSize2,
                                int trgSize1, int trgSize2,
                                Pair scale, Pair padding, Pair stride);

  template <class Op>
  friend void _applyRepMatOp(MatGPU& mat, const MatGPU& b, int dim, bool inner, Op op);
  friend void _varyColors(MatGPU &images, const MatGPU &add_mat);

};

#endif
