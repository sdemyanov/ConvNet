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

#include "mat_gpu.h"
#include "cuda_util.h"

void MatGPU::SetTensorDesc(const Dim &dims) {

  mexAssertMsg(size() == dims[0] * dims[1] * dims[2] * dims[3],
    "In SetTensorDesc dims assert");
  if (dims[0] > 0 || dims[1] > 0 || dims[2] > 0 || dims[3] > 0) {
    if (tensor_desc_ == NULL) {
      CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc_));
    }
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
      tensor_desc_, CUDNN_LAYOUT, CUDNN_TYPE,
      dims[0], dims[1], dims[2], dims[3]
    ));
  }
}

void MatGPU::SetFilterDesc(const Dim &dims) {
  mexAssertMsg(size() == dims[0] * dims[1] * dims[2] * dims[3],
    "In SetFilterDesc dims assert");
  if (dims[0] > 0 || dims[1] > 0 || dims[2] > 0 || dims[3] > 0) {
    if (filter_desc_ == NULL) {
      CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    }
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
      filter_desc_, CUDNN_TYPE, CUDNN_LAYOUT,
      dims[0], dims[1], dims[2], dims[3]
    ));
  }
}

cudnnTensorDescriptor_t MatGPU::GetTensorDesc() {
  mexAssertMsg(tensor_desc_ != NULL, "Empty tensor descriptor");
  return tensor_desc_;
}

cudnnFilterDescriptor_t MatGPU::GetFilterDesc() {
  mexAssertMsg(filter_desc_ != NULL, "Empty filter descriptor");
  return filter_desc_;
}

void MatGPU::ClearTensorDesc() {
  if (tensor_desc_ != NULL) {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc_));
    tensor_desc_ = NULL;
  }
}

void MatGPU::ClearFilterDesc() {
  if (filter_desc_ != NULL) {
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    filter_desc_ = NULL;
  }
}

// static

cudaEvent_t MatGPU::_start, MatGPU::_stop;

void MatGPU::StartCudaTimer() {
  if (print < 2) return;
  cudaEventRecord(_start, 0);
}

void MatGPU::MeasureCudaTime(std::string msg) {
  if (print < 2) return;
  float elapsedTime;
  cudaEventRecord(_stop, 0);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&elapsedTime, _start, _stop);
  mexPrintMsg(msg, elapsedTime);
}


curandGenerator_t MatGPU::_randGen;
cudaStream_t MatGPU::_defaultStream;
cublasHandle_t MatGPU::_cublasHandle;
cudnnHandle_t MatGPU::_cudnnHandle;
size_t MatGPU::_cudnnMemoryLimit;
MatGPU MatGPU::_workspace;


int MatGPU::getDeviceID() {
  int id;
  CUDA_CALL(cudaGetDevice(&id));
  return id;
}


void MatGPU::InitCuda(int gpu) {

  int num;
  CUDA_CALL(cudaGetDeviceCount(&num));
  mexAssertMsg(gpu < num, "Requested GPU index is not available");
  CUDA_CALL(cudaSetDevice(gpu));
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, getDeviceID()));
  mexPrintMsg("Executing on", prop.name);

  CUDA_CALL(cudaStreamCreate(&_defaultStream));
  CURAND_CALL(curandCreateGenerator(&_randGen, CURAND_RNG_PSEUDO_DEFAULT));
  CUBLAS_CALL(cublasCreate(&_cublasHandle));
  CUDNN_CALL(cudnnCreate(&_cudnnHandle));

  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);

}

void MatGPU::InitRand(int seed) {
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(_randGen, seed));
}

void MatGPU::CudaReset() {

  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);

  _workspace.clear();

  CUDNN_CALL(cudnnDestroy(_cudnnHandle));
  CUBLAS_CALL(cublasDestroy(_cublasHandle));
  CURAND_CALL(curandDestroyGenerator(_randGen));
  CUDA_CALL(cudaStreamDestroy(_defaultStream));

  CUDA_CALL(cudaDeviceReset());
}

void MatGPU::SetMemoryLimit(size_t memory) {
  // converting from megabytes to bytes
  _cudnnMemoryLimit = memory * 1024 * 1024;
}

// data access

ftype MatGPU::operator () (size_t i, size_t j) const {
  ftype *val_ptr = new ftype [1];
  CUDA_CALL(cudaMemcpy(val_ptr, &data(i, j), sizeof(ftype), cudaMemcpyDeviceToHost));
  ftype val = val_ptr[0];
  delete [] val_ptr;
  return val;
}
/*
ftype MatGPU::operator () (size_t ind) const {
  ftype *val_ptr = new ftype [1];
  CUDA_CALL(cudaMemcpy(val_ptr, &data(ind), sizeof(ftype), cudaMemcpyDeviceToHost));
  ftype val = val_ptr[0];
  delete [] val_ptr;
  return val;
}

MatGPU MatGPU::operator () (size_t ind) {
  MatGPU val_mat;
  val_mat.attach(&data(ind), 1, 1);
  return val_mat;
}
*/
size_t MatGPU::BytesNum() const {
  return size() * sizeof(ftype);
}

// memory functions

MatGPU::MatGPU() {
  init();
}

MatGPU::MatGPU(size_t size1, size_t size2) {
  init();
  resize(size1, size2);
}

MatGPU::MatGPU(const MatGPU &b) {
  init();
  if (b.empty()) return;
  resize(b.size1_, b.size2_);
  (*this) = b;
}

MatGPU::MatGPU(MatGPU &&b) {
  init();
  Swap(*this, b);
}

MatGPU::~MatGPU() {
  clear();
}

MatGPU& MatGPU::init() {
  data_ = NULL;
  size1_ = 0;
  size2_ = 0;
  order_ = kInternalOrder;
  owner_ = false;
  tensor_desc_ = NULL;
  filter_desc_ = NULL;
  return *this;
}

// copies only the content, not other parameters!!
MatGPU& MatGPU::operator = (const MatGPU &b) {
  //mexPrintMsg("Array assignment");
  mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
    "In MatGPU::operator = the arrays have different size");
  if (order_ == b.order_) {
    CUDA_CALL(cudaMemcpy(data_, b.data_, BytesNum(), cudaMemcpyDeviceToDevice));
  } else if (b.order_ != kInternalOrder) { // order_ == kInternalOrder
    MatGPU br;
    br.attach(b.data_, b.size2_, b.size1_, kInternalOrder);
    Trans(br, *this);
  } else { // b.order_ == kInternalOrder, order_ != kInternalOrder
    MatGPU br;
    br.attach(data_, size2_, size1_, kInternalOrder);
    Trans(b, br);
  }
  /*
  if (b.tensor_desc_ != NULL) {
    SetTensorDesc(b.tensor_shape());
  } else {
    tensor_desc_ = NULL;
  }
  if (b.filter_desc_ != NULL) {
    SetFilterDesc(b.filter_shape());
  } else {
    filter_desc_ = NULL;
  } */
  return *this;
}

MatGPU& MatGPU::resize(size_t size1, size_t size2) {
  // required for all cuda opearations
  mexAssertMsg(size1 <= INT_MAX / size2, "Matrix is too large!");
  if (size1 * size2 != size()) {
    clear();
    if (size1 * size2 > 0) {
      //mexPrintInt("rs1", size1);
      //mexPrintInt("rs2", size2);
      CUDA_CALL(cudaMalloc(&data_, size1 * size2 * sizeof(ftype)));
      owner_ = true;
    }
  }
  if (size1 != size1_ || size2 != size2_) {
    ClearTensorDesc();
    ClearFilterDesc();
  }
  size1_ = size1;
  size2_ = size2;
  return *this;
}

MatGPU& MatGPU::resize_tensor(Dim dims) {
  resize(dims[0], dims[1] * dims[2] * dims[3]);
  SetTensorDesc(dims);
  return *this;
}

MatGPU& MatGPU::resize_filter(Dim dims) {
  resize(dims[0], dims[1] * dims[2] * dims[3]);
  SetFilterDesc(dims);
  return *this;
}

MatGPU& MatGPU::reshape(size_t size1, size_t size2) {
  mexAssertMsg(size() == size1 * size2,
    "In MatGPU::reshape the sizes do not correspond");
  resize(size1, size2);
  return *this;
}

MatGPU& MatGPU::reshape_tensor(Dim dims) {
  mexAssertMsg(size() == dims[0] * dims[1] * dims[2] * dims[3],
    "In MatGPU::reshape_tensor the dimensions do not correspond");
  resize_tensor(dims);
  return *this;
}

MatGPU& MatGPU::reshape_filter(Dim dims) {
  mexAssertMsg(size() == dims[0] * dims[1] * dims[2] * dims[3],
    "In MatGPU::reshape_tensor the dimensions do not correspond");
  resize_filter(dims);
  return *this;
}

MatGPU& MatGPU::attach(const MatGPU &b) {
  return attach(b.data_, b.size1_, b.size2_, b.order_);
}

MatGPU& MatGPU::attach(const MatGPU &b, size_t offset, size_t size1, size_t size2, bool order) {
  //mexAssertMsg(b.size1_ == 1 || b.size2_ == 1, "In MatGPU::attach with offset one of sizes should be 1");
  mexAssertMsg(offset + size1 * size2 <= b.size(),
            "In MatGPU::attach the sizes don't correspond each other");
  return attach(b.data_ + offset, size1, size2, order);
}

MatGPU& MatGPU::attach(ftype *ptr, size_t size1, size_t size2) {
  return attach(ptr, size1, size2, kInternalOrder);
}

MatGPU& MatGPU::attach(ftype *ptr, size_t size1, size_t size2, bool order) {
  //mexAssertMsg(order == false, "In MatGPU::attach order should be always false");
  clear();
  data_ = ptr;
  size1_ = size1;
  size2_ = size2;
  order_ = order;
  return *this;
}

MatGPU& MatGPU::clear() {
  ClearTensorDesc();
  ClearFilterDesc();
  if (owner_) {
    mexAssert(data_ != NULL);
    mexAssert(size() > 0);
    //mexPrintInt("clear s1", size1_);
    //mexPrintInt("clear s2", size2_);
    CUDA_CALL(cudaFree(data_));
    owner_ = false;
  }
  init();
  //mexPrintMsg("Array clear end");
  return *this;
}

// be careful of using it as it does not guarantee
// that it is not used somewhere else at the same time
MatGPU& MatGPU::GetFromWorkspace(size_t size1, size_t size2) {
  if (size1 * size2 > MatGPU::_workspace.size()) {
    MatGPU::_workspace.resize(size1, size2);
  }
  attach(MatGPU::_workspace, 0, size1, size2, kInternalOrder);
  return *this;
}

void Swap(MatGPU &a, MatGPU &b) {

  ftype *data_tmp = b.data_;
  b.data_ = a.data_;
  a.data_ = data_tmp;

  size_t size1_tmp = b.size1_;
  b.size1_ = a.size1_;
  a.size1_ = size1_tmp;

  size_t size2_tmp = b.size2_;
  b.size2_ = a.size2_;
  a.size2_ = size2_tmp;

  bool order_tmp = b.order_;
  b.order_ = a.order_;
  a.order_ = order_tmp;

  bool owner_tmp = b.owner_;
  b.owner_ = a.owner_;
  a.owner_ = owner_tmp;

  cudnnTensorDescriptor_t tensor_desc_tmp_ = b.tensor_desc_;
  b.tensor_desc_ = a.tensor_desc_;
  a.tensor_desc_ = tensor_desc_tmp_;

  cudnnFilterDescriptor_t filter_desc_tmp_ = b.filter_desc_;
  b.filter_desc_ = a.filter_desc_;
  a.filter_desc_ = filter_desc_tmp_;
}

// data functions

MatGPU& MatGPU::ident() {
  cuda_ident(*this);
  return *this;
}

MatGPU& MatGPU::assign(ftype val) {
  cuda_assval(*this, val);
  return *this;
}

MatGPU& MatGPU::rand() {
  if (!empty()) {
    CURAND_CALL(curandGenerateUniform(_randGen, data_, size()));
  }
  return *this;
}

MatGPU& MatGPU::randnorm() {
  if (!empty()) {
    CURAND_CALL(curandGenerateNormal(_randGen, data_, size(), 0, 1));
  }
  return *this;
}

MatGPU& MatGPU::linear(ftype ca, ftype cb, const MatGPU &b, bool b_tr) {
  mexAssertMsg(cb == 0 || data_ != b.data_, "In linear pointers should be different");
  mexAssertMsg(order_ == b.order_, "In linear orders should be the same");
  cublasOperation_t a_op = CUBLAS_OP_N, b_op;
  if (!b_tr) {
    mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
      "In linear sizes does not correspond to each other");
    b_op = CUBLAS_OP_N;
  } else {
    mexAssertMsg(size1_ == b.size2_ && size2_ == b.size1_,
      "In linear sizes does not correspond to each other");
    b_op = CUBLAS_OP_T;
  }
  int as1, as2, bs1;
  if (order_ == false) {
    as1 = size1_; as2 = size2_;
    bs1 = b.size1_;
  } else {
    as1 = size2_; as2 = size1_;
    bs1 = b.size2_;
  }
  cudaStream_t stream = MatGPU::_defaultStream;
  cublasHandle_t handle = MatGPU::_cublasHandle;
  const ftype scale1 = ca, scale2 = cb;
  CUBLAS_CALL(cublasSetStream(handle, stream));
  CUBLAS_CALL(cublasSgeam(handle, a_op, b_op, as1, as2,
              &scale1, data_, as1,
              &scale2, b.data_, bs1,
              data_, as1));
  return *this;
}


MatGPU& MatGPU::operator += (const MatGPU &b) {
  cuda_addmat(*this, b);
  //linear(1, 1, b, false);
  return *this;
}

MatGPU& MatGPU::operator -= (const MatGPU &b) {
  cuda_submat(*this, b);
  //linear(1, -1, b, false);
  return *this;
}

MatGPU& MatGPU::operator *= (const MatGPU &b) {
  cuda_multmat(*this, b);
  return *this;
}

MatGPU& MatGPU::operator /= (const MatGPU &b) {
  cuda_divmat(*this, b);
  return *this;
}

MatGPU& MatGPU::operator += (ftype c) {
  cuda_addval(*this, c);
  return *this;
}

MatGPU& MatGPU::operator -= (ftype c) {
  cuda_subval(*this, c);
  return *this;
}

MatGPU& MatGPU::operator *= (ftype c) {
  cuda_multval(*this, c);
  //linear(c, 0, *this, false);
  return *this;
}

MatGPU& MatGPU::operator /= (ftype c) {
  cuda_divval(*this, c);
  //linear(1.0/c, 0, *this, false);
  return *this;
}

MatGPU& MatGPU::Sign() {
  cuda_sign(*this);
  return *this;
}

MatGPU& MatGPU::Sqrt() {
  cuda_sqrt(*this);
  return *this;
}

MatGPU& MatGPU::Log() {
  cuda_log(*this);
  return *this;
}

MatGPU& MatGPU::Exp() {
  cuda_exp(*this);
  return *this;
}

MatGPU& MatGPU::Sigmoid() {
  cuda_sigmoid(*this);
  return *this;
}

MatGPU& MatGPU::SigmDer(const MatGPU& a) {
  cuda_sigmder(*this, a);
  return *this;
}

MatGPU& MatGPU::SoftMax() {
  mexAssert(kInternalOrder == true);
  cudnnTensorDescriptor_t src_desc = GetTensorDesc();
  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnSoftmaxForward(
    MatGPU::_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    &scale_res, src_desc, data_, &scale_cur, src_desc, data_
  ));
  return *this;
}

MatGPU& MatGPU::SoftDer(const MatGPU& b) {
  //computeSoftmaxGrad(a, *this, *this);
  mexAssert(kInternalOrder == true);
  MatGPU bc;
  if (b.order_ == false) {
    MatGPU btr;
    btr.attach(b, 0, size2_, size1_, true);
    bc.resize(b.size1_, b.size2_);
    bc.set_order(true);
    Trans(btr, bc);
  } else {
    bc.attach(b);
  }
  cudnnTensorDescriptor_t src_desc = GetTensorDesc();
  cudnnTensorDescriptor_t par_desc = bc.GetTensorDesc();
  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnSoftmaxBackward(
    MatGPU::_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    &scale_res, par_desc, bc.data_, src_desc, data_, &scale_cur, src_desc, data_
  ));

  return *this;
}

MatGPU& MatGPU::CondAssign(const MatGPU &condmat, bool incase, ftype threshold, ftype a) {
  cuda_condassign(*this, condmat, incase, threshold, a);
  return *this;
}

MatGPU& MatGPU::CondAdd(const MatGPU &condmat, bool incase, ftype threshold, ftype a) {
  cuda_condadd(*this, condmat, incase, threshold, a);
  return *this;
}

MatGPU& MatGPU::CondMult(const MatGPU &condmat, bool incase, ftype threshold, ftype a) {
  cuda_condmult(*this, condmat, incase, threshold, a);
  return *this;
}

MatGPU& MatGPU::AddVect(MatGPU &vect, int dim) {
  //cuda_addvect(*this, vect, dim);
  mexAssertMsg(data_ != vect.data_, "In AddVect pointers should be different");
  if (dim == 1) {
    mexAssertMsg(vect.size1_ == 1 && vect.size2_ == size2_,
              "In AddVect the sizes don't correspond");
  } else if (dim == 2) {
    mexAssertMsg(vect.size1_ == size1_ && vect.size2_ == 1,
              "In AddVect the sizes don't correspond");
  } else {
    mexAssertMsg(false, "In MatGPU::AddVect the dimension parameter must be either 1 or 2");
  }
  Dim dims = {(int) size1_, (int) size2_, 1, 1};
  Dim vect_dims = {(int) vect.size1_, (int) vect.size2_, 1, 1};
  reshape_tensor(dims);
  vect.reshape_tensor(vect_dims);
  AddTensor(vect);
  return *this;
}


MatGPU& MatGPU::MultVect(const MatGPU &vect, int dim) {
  //cuda_multvect(*this, vect, dim);

  mexAssertMsg(data_ != vect.data_, "In MultVect pointers should be different");

  cudaStream_t stream = MatGPU::_defaultStream;
  cublasHandle_t handle = MatGPU::_cublasHandle;
  CUBLAS_CALL(cublasSetStream(handle, stream));

  int as1, as2;
  cublasSideMode_t side_mode;
  if (order_ == false) {
    as1 = size1_; as2 = size2_;
  } else {
    as1 = size2_; as2 = size1_;
  }

  if (dim == 1) {
    mexAssertMsg(vect.size1_ == 1 && vect.size2_ == size2_,
      "In MultVect the sizes don't correspond");
    if (order_ == true) {
      side_mode = CUBLAS_SIDE_LEFT;
    } else {
      side_mode = CUBLAS_SIDE_RIGHT;
    }
  } else if (dim == 2) {
    mexAssertMsg(vect.size1_ == size1_ && vect.size2_ == 1,
      "In MultVect the sizes don't correspond");
    if (order_ == true) {
      side_mode = CUBLAS_SIDE_RIGHT;
    } else {
      side_mode = CUBLAS_SIDE_LEFT;
    }
  } else {
    mexAssertMsg(false, "In MatGPU::MultVect the dimension parameter must be either 1 or 2");
  }
  CUBLAS_CALL(cublasSdgmm(handle, side_mode, as1, as2,
              data_, as1, vect.data_, 1, data_, as1));

  return *this;
}

MatGPU& MatGPU::Reorder(bool order) {
  //mexAssertMsg(order_ == order, "In MatGPU::reorder order should be the same");
  if (order_ != order) {
    if (size1_ > 1 && size2_ > 1) {
      MatGPU mr(size1_, size2_);
      mr.order_ = order;
      mr = (*this); // reorder
      order_ = order;
      (*this) = mr;
    } else {
      order_ = order;
    }
  }
  return *this;
}

MatGPU& MatGPU::ReorderMaps(bool cur_order, bool order) {
  if (cur_order != order) {
    mexAssertMsg(order_ == true, "In ReorderMaps the order should be true");
    std::vector< std::vector<MatGPU> > maps = InitMaps();
    for (size_t i = 0; i < maps.size(); ++i) {
      for (size_t j = 0; j < maps[i].size(); ++j) {
        // setting the correct order instead of default kInternalOrder
        maps[i][j].set_order(cur_order);
        // actual reordering
        maps[i][j].Reorder(order);
      }
    }
  }
  return *this;
}

MatGPU& MatGPU::Validate() {
  cuda_validate(*this);
  return *this;
}

// const functions

Dim MatGPU::tensor_shape() const {
  mexAssertMsg(tensor_desc_ != NULL, "Tensor descriptor is not defined");
  Dim shape, stride;
  cudnnDataType_t data_type = CUDNN_TYPE;
  CUDNN_CALL(cudnnGetTensor4dDescriptor(
    tensor_desc_, &data_type,
    &shape[0], &shape[1], &shape[2], &shape[3],
    &stride[0], &stride[1], &stride[2], &stride[3]
  ));
  return shape;
}

Dim MatGPU::filter_shape() const {
  mexAssertMsg(filter_desc_ != NULL, "Filter descriptor is not defined");
  Dim shape;
  cudnnDataType_t data_type = CUDNN_TYPE;
  cudnnTensorFormat_t tensor_format = CUDNN_LAYOUT;
  CUDNN_CALL(cudnnGetFilter4dDescriptor(
    filter_desc_, &data_type, &tensor_format,
    &shape[0], &shape[1], &shape[2], &shape[3]
  ));
  return shape;
}

std::vector< std::vector<MatGPU> > MatGPU::InitMaps() const {
  mexAssertMsg(order_ == true, "In InitMaps the order should be true");
  mexAssertMsg(tensor_desc_ == NULL || filter_desc_ == NULL, "Both descriptors are defined");
  Dim dims;
  if (tensor_desc_ != NULL) {
    dims = tensor_shape();
  } else if (filter_desc_ != NULL) {
    dims = filter_shape();
  } else {
    mexAssertMsg(false, "Neither of descriptors is defined");
  }
  mexAssertMsg(size() == dims[0] * dims[1] * dims[2] * dims[3],
    "In InitMaps dims assert");
  // splitting the 2nd dimension
  size_t batchsize = dims[0], channels = dims[1], numel = dims[2] * dims[3];
  size_t pixels_num = channels * numel;
  std::vector< std::vector<MatGPU> > matrices(batchsize);
  for (size_t k = 0; k < batchsize; ++k) {
    matrices[k].resize(channels);
    for (size_t j = 0; j < channels; ++j) {
      matrices[k][j].attach(data_ + k * pixels_num + j * numel,
          dims[2], dims[3], kInternalOrder);
    }
  }
  return matrices;
}

ftype MatGPU::sum() const {
  return cuda_sum(*this);
}

// CPU <-> GPU functions

MatGPU& MatGPU::operator = (const MatCPU &a) {
  // no resize in order to ensure that b.data_ is fixed
  mexAssertMsg(!a.empty() && !empty(), "In HostToDevice one of the arrays is empty");
  mexAssertMsg(a.size1() == size1_ && a.size2() == size2_,
    "In HostToDevice the sizes of matrices do not correspond");
  // conversion is to get access to protected members
  const MatGPU *a_ptr = static_cast<const MatGPU*>(&a);
  if (a.order() == order_) {
    CUDA_CALL(cudaMemcpy(data_, a_ptr->data_, BytesNum(), cudaMemcpyHostToDevice));
  } else {
    MatGPU br(a.size1(), a.size2());
    br.order_ = a.order();
    br = a; // previous case
    (*this) = br; // reorder
  }
  return *this;
}

void DeviceToHost(const MatGPU &b, MatCPU &a) {
  // no resize in order to ensure that b.data_ is fixed
  mexAssertMsg(!a.empty() && !b.empty(), "In DeviceToHost one of the arrays is empty");
  mexAssertMsg(a.size1() == b.size1_ && a.size2() == b.size2_,
    "In DeviceToHost the sizes of matrices do not correspond");
  // conversion is to get access to protected members
  MatGPU *a_ptr = static_cast<MatGPU*>(&a);
  if (a.order() == b.order()) {
    CUDA_CALL(cudaMemcpy(a_ptr->data_, b.data_, b.BytesNum(), cudaMemcpyDeviceToHost));
  } else {
    MatGPU br(a.size1(), a.size2());
    br.order_ = a.order();
    br = b; // reorder
    a = br; // previous case
  }
}

void SubSet(MatCPU &a, MatGPU &b, size_t offset, bool dir) {

  if (print >= 3) {
    MatGPU::StartCudaTimer();
  }

  MatGPU *a_ptr = static_cast<MatGPU*>(&a);
  mexAssertMsg(a_ptr->order_ == true, "In SubSet 'a.order_' should be true");
  mexAssertMsg(b.order_ == true, "In SubSet 'b.order_' should be true");
  mexAssertMsg(offset + b.size1_ <= a_ptr->size1_ && b.size2_ == a_ptr->size2_,
            "In SubSet the sizes don't correspond each other");
  MatCPU as;
  as.attach(a_ptr->data_ + offset * a_ptr->size2_, b.size1_, b.size2_, true);
  if (dir) {
    b = as; // HostToDevice
  } else {
    DeviceToHost(b, as);
  }
  if (print >= 3) {
    MatGPU::MeasureCudaTime("SubSet");
  }
}

// friend functions

void Sum(MatGPU &a, MatGPU &vect, int dim) {
  //cuda_sumvect(a, vect, dim);

  mexAssertMsg(a.data_ != vect.data_, "In Sum pointers should be different");

  cudaStream_t stream = MatGPU::_defaultStream;
  cublasHandle_t handle = MatGPU::_cublasHandle;
  CUBLAS_CALL(cublasSetStream(handle, stream));
  const ftype scale1 = 1.0, scale2 = 0.0;

  cublasOperation_t op;
  MatGPU ones_vect;
  if (dim == 1) {
    mexAssertMsg(vect.size1_ == 1 && vect.size2_ == a.size2_,
      "In Sum the sizes do not correspond each other");
    if (a.order_ == false) {
      op = CUBLAS_OP_T;
    } else {
      op = CUBLAS_OP_N;
    }
    ones_vect.GetFromWorkspace(1, a.size1_);

  } else if (dim == 2) {
    mexAssertMsg(vect.size1_ == a.size1_ && vect.size2_ == 1,
      "In Sum the sizes do not correspond each other");
    if (a.order_ == false) {
      op = CUBLAS_OP_N;
    } else {
      op = CUBLAS_OP_T;
    }
    ones_vect.GetFromWorkspace(a.size2_, 1);
  } else {
    mexAssertMsg(false, "In MatGPU::Sum the dimension parameter must be either 1 or 2");
  }
  int as1, as2;
  if (a.order_ == false) {
    as1 = a.size1_;
    as2 = a.size2_;
  } else {
    as1 = a.size2_;
    as2 = a.size1_;
  }
  ones_vect.assign(1);
  CUBLAS_CALL(cublasSgemv(handle, op, as1, as2,
              &scale1, a.data_, as1,
              ones_vect.data_, 1,
              &scale2, vect.data_, 1));
}


void Mean(MatGPU &a, MatGPU &vect, int dim) {
  Sum(a, vect, dim);
  if (dim == 1) {
    vect /= (ftype) a.size1_;
  } else if (dim == 2) {
    vect /= (ftype) a.size2_;
  } else {
    mexAssertMsg(false, "In MatGPU::Mean the dimension parameter must be either 1 or 2");
  }
}

void Trans(const MatGPU &a, MatGPU &b) {
  //cuda_trans(a, b);
  b.linear(0, 1, a, true);
}

// layer transformation functions

void Prod(const MatGPU &a, bool a_tr, const MatGPU &b, bool b_tr, MatGPU &c) {

  mexAssertMsg(a.order_ == b.order_ && b.order_ == c.order_, "In Prod the orders should be the same");

  cudaStream_t stream = MatGPU::_defaultStream;
  cublasHandle_t handle = MatGPU::_cublasHandle;
  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUBLAS_CALL(cublasSetStream(handle, stream));

  cublasOperation_t a_op, b_op;
  if (!a_tr) {
    a_op = CUBLAS_OP_N;
  } else {
    a_op = CUBLAS_OP_T;
  }
  if (!b_tr) {
    b_op = CUBLAS_OP_N;
  } else {
    b_op = CUBLAS_OP_T;
  }

  int as1, as2, bs1, bs2;
  if (a.order_ == false) { // Alex kernels
    if (!a_tr) { // a
      as1 = a.size1_; as2 = a.size2_;
    } else { // aT
      as1 = a.size2_; as2 = a.size1_;
    }
    if (!b_tr) { // b
      bs1 = b.size1_; bs2 = b.size2_;
    } else { // bT
      bs1 = b.size2_; bs2 = b.size1_;
    }
    mexAssertMsg(as2 == bs1, "In Prod the sizes of matrices do not correspond");
    mexAssertMsg(c.size1_ == as1 && c.size2_ == bs2, "In Prod the size of output matrix is wrong");

    CUBLAS_CALL(cublasSgemm(handle, a_op, b_op, as1, bs2, as2,
                           &scale_res, a.data_, a.size1_, b.data_, b.size1_,
                           &scale_cur, c.data_, c.size1_));
  } else { // cuDNN kernels
    if (!a_tr) { // a
      as1 = a.size2_; as2 = a.size1_;
    } else { // aT
      as1 = a.size1_; as2 = a.size2_;
    }
    if (!b_tr) { // b
      bs1 = b.size2_; bs2 = b.size1_;
    } else { // bT
      bs1 = b.size1_; bs2 = b.size2_;
    }
    mexAssertMsg(as1 == bs2, "In Prod the sizes of matrices do not correspond");
    mexAssertMsg(c.size1_ == as2 && c.size2_ == bs1, "In Prod the size of output matrix is wrong");

    CUBLAS_CALL(cublasSgemm(handle, b_op, a_op, bs1, as2, bs2,
                           &scale_res, b.data_, b.size2_, a.data_, a.size2_,
                           &scale_cur, c.data_, c.size2_));
  }
}

// filter functions

MatGPU& MatGPU::AddTensor(MatGPU &tensor) {
  mexAssert(kInternalOrder == true);

  cudnnTensorDescriptor_t desc = GetTensorDesc();
  cudnnTensorDescriptor_t tns_desc = tensor.GetTensorDesc();

  const ftype scale_res = 1.0, scale_cur = 1.0;
  CUDNN_CALL(cudnnAddTensor(MatGPU::_cudnnHandle,
    &scale_res, tns_desc, tensor.data_,
    &scale_cur, desc, data_
  ));
  return *this;
}

void ConvolutionForward(MatGPU& activs, MatGPU& filters, MatGPU& targets,
                        const cudnnConvolutionDescriptor_t &conv_desc) {
  mexAssert(kInternalOrder == true);

  if (print >= 3) {
    MatGPU::StartCudaTimer();
  }
  cudnnTensorDescriptor_t act_desc = activs.GetTensorDesc();
  cudnnFilterDescriptor_t flt_desc = filters.GetFilterDesc();
  cudnnTensorDescriptor_t trg_desc = targets.GetTensorDesc();
  Dim dims;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
    conv_desc, act_desc, flt_desc, &dims[0], &dims[1], &dims[2], &dims[3]
  ));
  Dim trg_shape = targets.tensor_shape();
  mexAssertMsg(trg_shape[0] == dims[0] && trg_shape[1] == dims[1] &&
               trg_shape[2] == dims[2] && trg_shape[3] == dims[3],
               "ConvolutionForward shape assert");

  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
    MatGPU::_cudnnHandle, act_desc, flt_desc, conv_desc, trg_desc,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
    MatGPU::_cudnnMemoryLimit, &algo
  ));

  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
    MatGPU::_cudnnHandle, act_desc, flt_desc, conv_desc, trg_desc,
    algo, &ws_size
  ));

  MatGPU workspace;
  workspace.GetFromWorkspace(1, ws_size / sizeof(ftype));

  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnConvolutionForward(MatGPU::_cudnnHandle,
    &scale_res, act_desc, activs.data_, flt_desc, filters.data_,
    conv_desc, algo, workspace.data_, ws_size,
    &scale_cur, trg_desc, targets.data_
  ));

  if (print >= 3) {
    MatGPU::MeasureCudaTime("FilterActs");
  }
}


void ConvolutionBackwardData(MatGPU& derivs, MatGPU& filters, MatGPU& targets,
                             const cudnnConvolutionDescriptor_t &conv_desc) {
  mexAssert(kInternalOrder == true);

  if (print >= 3) {
    MatGPU::StartCudaTimer();
  }
  cudnnTensorDescriptor_t trg_desc = targets.GetTensorDesc();
  cudnnFilterDescriptor_t flt_desc = filters.GetFilterDesc();
  cudnnTensorDescriptor_t der_desc = derivs.GetTensorDesc();
  Dim dims;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
    conv_desc, trg_desc, flt_desc, &dims[0], &dims[1], &dims[2], &dims[3]
  ));
  Dim der_shape = derivs.tensor_shape();
  mexAssertMsg(der_shape[0] == dims[0] && der_shape[1] == dims[1] &&
               der_shape[2] == dims[2] && der_shape[3] == dims[3],
               "ConvolutionBackwardData shape assert");

  cudnnConvolutionBwdDataAlgo_t algo;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
    MatGPU::_cudnnHandle, flt_desc, der_desc, conv_desc, trg_desc,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
    MatGPU::_cudnnMemoryLimit, &algo
  ));

  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    MatGPU::_cudnnHandle, flt_desc, der_desc, conv_desc, trg_desc,
    algo, &ws_size
  ));

  MatGPU workspace;
  workspace.GetFromWorkspace(1, ws_size / sizeof(ftype));

  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnConvolutionBackwardData(MatGPU::_cudnnHandle,
    &scale_res, flt_desc, filters.data_, der_desc, derivs.data_,
    conv_desc, algo, workspace.data_, ws_size,
    &scale_cur, trg_desc, targets.data_
  ));

  if (print >= 3) {
    MatGPU::MeasureCudaTime("ImgActs");
  }
}


void ConvolutionBackwardFilter(MatGPU& activs, MatGPU& derivs, MatGPU& targets,
                               const cudnnConvolutionDescriptor_t &conv_desc) {
  mexAssert(kInternalOrder == true);

  cudnnTensorDescriptor_t act_desc = activs.GetTensorDesc();
  cudnnFilterDescriptor_t trg_desc = targets.GetFilterDesc();
  cudnnTensorDescriptor_t der_desc = derivs.GetTensorDesc();
  Dim dims;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
    conv_desc, act_desc, trg_desc, &dims[0], &dims[1], &dims[2], &dims[3]
  ));
  Dim der_shape = derivs.tensor_shape();
  mexAssertMsg(der_shape[0] == dims[0] && der_shape[1] == dims[1] &&
               der_shape[2] == dims[2] && der_shape[3] == dims[3],
               "ConvolutionBackwardFilter shape assert");

  cudnnConvolutionBwdFilterAlgo_t algo;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
    MatGPU::_cudnnHandle, act_desc, der_desc, conv_desc, trg_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
    MatGPU::_cudnnMemoryLimit, &algo
  ));

  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    MatGPU::_cudnnHandle, act_desc, der_desc, conv_desc, trg_desc,
    algo, &ws_size
  ));

  MatGPU workspace;
  workspace.GetFromWorkspace(1, ws_size / sizeof(ftype));

  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnConvolutionBackwardFilter(MatGPU::_cudnnHandle,
    &scale_res, act_desc, activs.data_, der_desc, derivs.data_,
    conv_desc, algo, workspace.data_, ws_size,
    &scale_cur, trg_desc, targets.data_
  ));
}


void ConvolutionBackwardBias(MatGPU& derivs, MatGPU &targets) {
  mexAssert(kInternalOrder == true);

  cudnnTensorDescriptor_t der_desc = derivs.GetTensorDesc();
  cudnnTensorDescriptor_t trg_desc = targets.GetTensorDesc();

  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnConvolutionBackwardBias(MatGPU::_cudnnHandle,
    &scale_res, der_desc, derivs.data_,
    &scale_cur, trg_desc, targets.data_
  ));
};

// scaling functions
void Pooling(MatGPU& activs, MatGPU& targets,
             cudnnPoolingDescriptor_t pool_desc) {

  mexAssert(kInternalOrder == true);
  cudnnTensorDescriptor_t act_desc = activs.GetTensorDesc();
  cudnnTensorDescriptor_t trg_desc = targets.GetTensorDesc();
  Dim dims;
  CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
    pool_desc, act_desc, &dims[0], &dims[1], &dims[2], &dims[3]
  ));
  Dim trg_shape = targets.tensor_shape();
  mexAssertMsg(trg_shape[0] == dims[0] && trg_shape[1] == dims[1] &&
            trg_shape[2] == dims[2] && trg_shape[3] == dims[3],
            "Pooling shape assert");

  const ftype scale_res = 1.0, scale_cur = 0.0;
  CUDNN_CALL(cudnnPoolingForward(MatGPU::_cudnnHandle, pool_desc,
    &scale_res, act_desc, activs.data_,
    &scale_cur, trg_desc, targets.data_
  ));
}


void PoolingUndo(MatGPU& activs, MatGPU& pool_activs,
                 MatGPU& pool_derivs, MatGPU& derivs,
                 cudnnPoolingDescriptor_t pool_desc, bool dir) {
  // dir == true -> backward, derivs are targets
  // dir == false -> forward, pool_derivs are targets
  mexAssert(kInternalOrder == true);
  mexAssertMsg(activs.size1_ == derivs.size1_ &&
               activs.size2_ == derivs.size2_,
            "In 'PoolingUndo' activs size assert");
  mexAssertMsg(pool_activs.size1_ == pool_derivs.size1_ &&
            pool_activs.size2_ == pool_derivs.size2_,
            "In 'PoolingUndo' pool_activs.size assert");
  cudnnTensorDescriptor_t act_desc = activs.GetTensorDesc();
  cudnnTensorDescriptor_t trg_desc = derivs.GetTensorDesc();
  cudnnTensorDescriptor_t pool_act_desc = pool_activs.GetTensorDesc();
  cudnnTensorDescriptor_t pool_der_desc = pool_derivs.GetTensorDesc();
  Dim dims;
  CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
    pool_desc, act_desc, &dims[0], &dims[1], &dims[2], &dims[3]
  ));
  Dim pool_act_shape = pool_activs.tensor_shape();
  mexAssertMsg(pool_act_shape[0] == dims[0] && pool_act_shape[1] == dims[1] &&
            pool_act_shape[2] == dims[2] && pool_act_shape[3] == dims[3],
            "PoolingUndo shape assert");

  if (dir == true) {
    const ftype scale_res = 1.0, scale_cur = 0.0;
    CUDNN_CALL(cudnnPoolingBackward(MatGPU::_cudnnHandle, pool_desc,
      &scale_res, pool_act_desc, pool_activs.data_,
      pool_der_desc, pool_derivs.data_, act_desc, activs.data_,
      &scale_cur, trg_desc, derivs.data_
    ));
  } else {
    Dim prev_dims = activs.tensor_shape();
    Pair scale, padding, stride;
    cudnnPoolingMode_t pool_mode = CUDNN_POOLING_MAX;
    cudnnNanPropagation_t nan_prop_mode = CUDNN_PROPAGATE_NAN;
    CUDNN_CALL(cudnnGetPooling2dDescriptor(pool_desc,
      &pool_mode, &nan_prop_mode,
      &scale[0], &scale[1], &padding[0], &padding[1], &stride[0], &stride[1]
    ));
    _maxPoolThirdPass(activs, pool_activs, derivs, pool_derivs,
                      prev_dims[2], prev_dims[3], dims[2], dims[3],
                      scale, padding, stride);
  }
}


void AffineTransform(const MatGPU &images, MatGPU &targets,
                   const MatGPU &shift_mat, const MatGPU &scale_mat,
                   const MatGPU &mirror_mat, const MatGPU &angle_mat,
                   ftype defval, bool dir) {
  Dim img_dims = images.tensor_shape();
  Dim trg_dims = targets.tensor_shape();

  _affineTransform(images, targets,
                 img_dims[2], img_dims[3], trg_dims[2], trg_dims[3],
                 shift_mat, scale_mat, mirror_mat, angle_mat, defval, dir);
}

/*

void VaryColors(MatGPU &images, const Dim &dims,
                const MatGPU &eigenvectors, ftype noise_std) {

  int batchsize = images.size1();
  int channels = images.size2() / (dims[2] * dims[3]);

  MatGPU noise_mat, add_mat;
  MatGPU::swapWithBuffer(noise_mat, -7);
  noise_mat.resize(batchsize, channels);
  // hack, because randnorm does not work for odd numbers. Start.
  if (noise_mat.size1() * noise_mat.size2() % 2 > 0) {
    MatGPU rndmat;
    rndmat.attach(
      noise_mat, 0,
      noise_mat.size1() * noise_mat.size2() - 1, 1, noise_mat.order()
    );
    rndmat.randnorm() *= noise_std;

    rndmat.attach(
      noise_mat, noise_mat.size1() * noise_mat.size2() - 1,
      1, 1, noise_mat.order()
    );
    (rndmat.rand() -= 0.5) *= noise_std;
  } else {
    noise_mat.randnorm() *= noise_std;
  }
  // hack, because randnorm does not work for odd numbers. End.

  MatGPU::swapWithBuffer(add_mat, -8);
  add_mat.resize(batchsize, channels);
  Prod(noise_mat, false, eigenvectors, true, add_mat);

  _varyColors(images, add_mat);

  MatGPU::swapWithBuffer(noise_mat, -7);
  MatGPU::swapWithBuffer(add_mat, -8);
} */

