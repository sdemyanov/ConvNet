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
   
#include "mat_gpu.h"
#include "cuda_util.h"

#ifndef CUDA_CALL
	#define CUDA_CALL(fun) { \
    int my_err_code = (fun); \
    if (my_err_code != cudaSuccess) { \
      char errmsg[100];\
      sprintf(errmsg, "CUDA function call failed![%d] %s:%d", my_err_code, __FILE__, __LINE__);\
      mexAssert(false, errmsg); \
    } \
  }
#endif

#ifndef CURAND_CALL
	#define CURAND_CALL(fun) { \
    int my_err_code = (fun); \
    if (my_err_code != CURAND_STATUS_SUCCESS) { \
      char errmsg[100];\
      sprintf(errmsg, "CURAND function call failed![%d] %s:%d", my_err_code, __FILE__, __LINE__);\
      mexAssert(false, errmsg); \
    } \
  }
#endif

#ifndef CUBLAS_CALL
	#define CUBLAS_CALL(fun) { \
    int my_err_code = (fun); \
    if (my_err_code != CUBLAS_STATUS_SUCCESS) { \
      char errmsg[100];\
      sprintf(errmsg, "CUBLAS function call failed![%d] %s:%d", my_err_code, __FILE__, __LINE__);\
      mexAssert(false, errmsg); \
    } \
  }
#endif

#ifndef CUDNN_CALL
	#define CUDNN_CALL(fun) { \
    int my_err_code = (fun); \
    if (my_err_code != CUDNN_STATUS_SUCCESS) { \
      char errmsg[100];\
      sprintf(errmsg, "CUDNN function call failed![%d] %s:%d", my_err_code, __FILE__, __LINE__);\
      mexAssert(false, errmsg); \
    } \
  }
#endif

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

std::map<int, MatGPU> MatGPU::_buffers;

void MatGPU::swapWithBuffer(MatGPU &mat, int key) {
  std::map<int, MatGPU>::iterator it = _buffers.find(key);  
  if (it == _buffers.end()) {
    //mexPrintMsg("key", key);
    _buffers[key] = MatGPU();
    it = _buffers.find(key);
  }
  Swap(mat, it->second);
}

curandGenerator_t MatGPU::_randGen;
cudaStream_t MatGPU::_defaultStream;
cublasHandle_t MatGPU::_cublasHandle;
#if USE_CUDNN == 1
  cudnnHandle_t MatGPU::_cudnnHandle;
#endif


int MatGPU::getDeviceID() {
  int id;
  CUDA_CALL(cudaGetDevice(&id));  
  return id;
}

void MatGPU::CudaInit() {

  int num;  
  mexAssert(cudaGetDeviceCount(&num) == cudaSuccess, "No proper CUDA device is found");
  
  #if USE_CUDNN == 1
    //CUDNN_CALL(cudnnCreate(&_cudnnHandle));
  #endif
  CUDA_CALL(cudaStreamCreate(&_defaultStream));  
  CURAND_CALL(curandCreateGenerator(&_randGen, CURAND_RNG_PSEUDO_DEFAULT));
  CUBLAS_CALL(cublasCreate(&_cublasHandle));  
  
  /*
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, getDeviceID()));
  mexPrintMsg("Executing on", prop.name);    
  */
  
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);  
  
}

void MatGPU::InitRand(size_t seed) {  
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(_randGen, seed));
}

void MatGPU::CudaReset() {

  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
  
  _buffers.clear();

  #if USE_CUDNN == 1
    //CUDNN_CALL(cudnnDestroy(_cudnnHandle));
  #endif
  CUBLAS_CALL(cublasDestroy(_cublasHandle));
  CURAND_CALL(curandDestroyGenerator(_randGen));  
  CUDA_CALL(cudaStreamDestroy(_defaultStream));  
   
  CUDA_CALL(cudaDeviceReset());
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
size_t MatGPU::getNumDataBytes() const {
  return size1_ * size2_ * sizeof(ftype);
}

cudaTextureObject_t MatGPU::getTextureObject() {
  if (texture_ == 0) {    
    ftype *data_ptr;
    if (owner_) {
      data_ptr = data_;
    } else {      
      // for some reason textures are not created for non-owner pointers
      CUDA_CALL(cudaMalloc(&textdata_, getNumDataBytes()));      
      data_ptr = textdata_;
    }
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = data_ptr;
    resDesc.res.linear.sizeInBytes = getNumDataBytes();    
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    CUDA_CALL(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, NULL));    
    /*
    mexPrintInt("texture", (int) texture_);
    CUDA_CALL(cudaDestroyTextureObject(texture_));    
    mexAssert(false, "Success!");
    */
  }
  if (!owner_) {    
    CUDA_CALL(cudaMemcpy(textdata_, data_, getNumDataBytes(), cudaMemcpyDeviceToDevice));
  }  
  return texture_;
}

// memory functions

MatGPU::MatGPU() {
  //mexPrintMsg("Array constructor 0");
  init();
  //mexPrintMsg("Array constructor 0 end");
}

MatGPU::MatGPU(const std::vector<size_t> &newsize) {
  //mexPrintMsg("Array constructor 2");
  mexAssert(newsize.size() == 2, "In MatGPU::MatGPU the size vector length != 2");  
  init();
  resize(newsize[0], newsize[1]);
  //mexPrintMsg("Array constructor 2 end");
}

MatGPU::MatGPU(size_t size1, size_t size2) {
  //mexPrintMsg("Array constructor 2");
  init();
  resize(size1, size2);
  //mexPrintMsg("Array constructor 2 end");
}

MatGPU::MatGPU(const MatGPU &a) {
  //mexPrintMsg("Array copy constructor");
  init();  
  if (a.empty()) return;
  resize(a.size1_, a.size2_);
  (*this) = a;  
  //mexPrintMsg("Array copy constructor end");
}

MatGPU::MatGPU(MatGPU &&a) {
  //mexPrintMsg("Array move constructor");   
  init();
  Swap(*this, a);
  //mexPrintMsg("Array move constructor end");
}

MatGPU::~MatGPU() {
  clear();
}

MatGPU& MatGPU::init() {
  data_ = NULL;
  size1_ = 0;
  size2_ = 0;  
  stride_ = 1;
  order_ = kDefaultOrder;
  owner_ = false;
  textdata_ = NULL;
  texture_ = 0;
  return *this;
}

MatGPU& MatGPU::operator = (const MatGPU &a) {
  //mexPrintMsg("Array assignment");  
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In MatGPU::operator = the arrays have different size");
  mexAssert(stride_ == 1 && a.stride_ == 1,
    "In MatGPU::operator = the strides should be 1");
  if (order_ == a.order_) {
    CUDA_CALL(cudaMemcpy(data_, a.data_, getNumDataBytes(), cudaMemcpyDeviceToDevice));      
  } else if (a.order_ != kDefaultOrder) { // order_ == kDefaultOrder    
    MatGPU ar;
    ar.attach(a.data_, a.size2_, a.size1_, a.stride_, kDefaultOrder);    
    Trans(ar, *this);
  } else { // a.order_ == kDefaultOrder, order_ == !kDefaultOrder      
    MatGPU ar;
    ar.attach(data_, size2_, size1_, stride_, kDefaultOrder);    
    Trans(a, ar);
  }  
  //mexPrintMsg("Array assignment end");
  return *this;  
}

MatGPU& MatGPU::resize(size_t size1, size_t size2) {
  //mexPrintMsg("Array resize");
  if (size1 * size2 != size1_ * size2_) {
    clear();
    if (size1 * size2 > 0) {       
      CUDA_CALL(cudaMalloc(&data_, size1 * size2 * sizeof(ftype)));
      owner_ = true;
    }
  }
  size1_ = size1;
  size2_ = size2;  
  //mexPrintMsg("Array resize end");  
  return *this;
}

MatGPU& MatGPU::reshape(size_t size1, size_t size2) {
  mexAssert(size1_ * size2_ == size1 * size2,
    "In MatCPU::reshape the sizes do not correspond");
  size1_ = size1;
  size2_ = size2;
  return *this;
}

MatGPU& MatGPU::reorder(bool order, bool real) {
  //mexAssert(order_ == order, "In MatGPU::reorder order should be the same");
  if (order_ != order) {
    if (real == true) {
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

MatGPU& MatGPU::attach(const MatGPU &a) {
  return attach(a.data_, a.size1_, a.size2_, a.stride_, a.order_);
}

MatGPU& MatGPU::attach(const MatGPU &a, size_t offset, size_t size1, size_t size2, bool order) {  
  mexAssert(a.size1_ == 1 || a.size2_ == 1, "In MatGPU::attach with offset one of sizes should be 1");
  mexAssert(offset + size1 * size2 <= a.size1_ * a.size2_, 
            "In MatGPU::attach the sizes don't correspond each other");
  return attach(a.data_ + offset, size1, size2, a.stride_, order);    
}

MatGPU& MatGPU::attach(ftype *ptr, size_t size1, size_t size2) {
  return attach(ptr, size1, size2, 1, kDefaultOrder);
}

MatGPU& MatGPU::attach(ftype *ptr, size_t size1, size_t size2, size_t stride, bool order) {  
  //mexAssert(order == false, "In MatGPU::attach order should be always false");  
  clear();
  data_ = ptr;
  size1_ = size1;
  size2_ = size2;  
  stride_ = stride;    
  order_ = order;
  return *this;
}

MatGPU& MatGPU::clear() {
  //mexPrintMsg("Array clear");
  if (texture_ != 0) {
    CUDA_CALL(cudaDestroyTextureObject(texture_));    
    texture_ = 0;
    if (textdata_ != NULL) {
      CUDA_CALL(cudaFree(textdata_));
      textdata_ = NULL;    
    }
  }
  if (owner_) {
    CUDA_CALL(cudaFree(data_));
    owner_ = false;    
  }
  //mexPrintMsg("Array clear end");
  return init();
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
  
  size_t stride_tmp = b.stride_;
  b.stride_ = a.stride_;
  a.stride_ = stride_tmp;
    
  bool order_tmp = b.order_;
  b.order_ = a.order_;
  a.order_ = order_tmp;  
  
  bool owner_tmp = b.owner_;
  b.owner_ = a.owner_;
  a.owner_ = owner_tmp;  

  ftype *textdata_tmp = b.textdata_;
  b.textdata_ = a.textdata_;
  a.textdata_ = textdata_tmp;
  
  cudaTextureObject_t texObj_tmp = b.texture_;
  b.texture_ = a.texture_;
  a.texture_ = texObj_tmp;  
}

// data functions

MatGPU& MatGPU::assign(ftype val) {
  cuda_assval(*this, val);
  return *this;
}

MatGPU& MatGPU::rand() {
  mexAssert(stride_ == 1, "In MatGPU::rand stride_ should be 1");
  CURAND_CALL(curandGenerateUniform(_randGen, data_, size1_ * size2_));
  return *this;
}

MatGPU& MatGPU::randnorm() {
  mexAssert(stride_ == 1, "In MatGPU::randnorm stride_ should be 1");
  CURAND_CALL(curandGenerateNormal(_randGen, data_, size1_ * size2_, 0, 1));
  return *this;
}

MatGPU& MatGPU::operator += (const MatGPU &a) {
  cuda_addmat(*this, a);
  return *this;
}

MatGPU& MatGPU::operator -= (const MatGPU &a) {
  cuda_submat(*this, a);
  return *this;
}

MatGPU& MatGPU::operator *= (const MatGPU &a) {
  cuda_multmat(*this, a);
  return *this;
}

MatGPU& MatGPU::operator /= (const MatGPU &a) {
  cuda_divmat(*this, a);
  return *this;
}

MatGPU& MatGPU::operator += (ftype a) {
  cuda_addval(*this, a);
  return *this;
}

MatGPU& MatGPU::operator -= (ftype a) {
  cuda_subval(*this, a);
  return *this;
}

MatGPU& MatGPU::operator *= (ftype a) {
  cuda_multval(*this, a);
  return *this;
}

MatGPU& MatGPU::operator /= (ftype a) {
  cuda_divval(*this, a);
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
  MatGPU maxvect, sumvect;  
  swapWithBuffer(maxvect, -1);
  maxvect.resize(size1_, 1);
  cuda_maxvect(*this, maxvect, 2);
  this->AddVect(maxvect *= -1, 2);
  this->Exp();
  swapWithBuffer(sumvect, -2);  
  sumvect.resize(size1_, 1);
  Sum(*this, sumvect, 2);  
  maxvect.assign(1);
  maxvect /= sumvect;
  this->MultVect(maxvect, 2);
  swapWithBuffer(maxvect, -1);
  swapWithBuffer(sumvect, -2);
  return *this;
}

MatGPU& MatGPU::SoftDer(const MatGPU& a) {
  computeSoftmaxGrad(a, *this, *this);
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

MatGPU& MatGPU::AddVect(const MatGPU &vect, size_t dim) {  
  cuda_addvect(*this, vect, dim);
  return *this;
}

MatGPU& MatGPU::MultVect(const MatGPU &vect, size_t dim) {  
  cuda_multvect(*this, vect, dim);
  return *this;
}

MatGPU& MatGPU::Normalize(ftype norm) {
  mexAssert(false, "All normalization should be on the CPU");  
  return *this;
}

MatGPU& MatGPU::Validate() {    
  cuda_validate(*this);
  //CUDA_CALL(cudaDeviceSynchronize());
  return *this;
}

// CPU <-> GPU functions

MatGPU& MatGPU::operator = (const MatCPU &a) {
  // no resize in order to ensure that b.data_ is fixed
  mexAssert(!a.empty() && !empty(), "In HostToDevice one of the arrays is empty");
  mexAssert(a.size1() == size1_ && a.size2() == size2_,
    "In HostToDevice the sizes of matrices do not correspond");  
  // conversion is to get access to protected members
  const MatGPU *a_ptr = static_cast<const MatGPU*>(&a);
  mexAssert(a_ptr->stride_ == 1 && stride_ == 1, "In HostToDevice strides should be 1");
  if (a.order() == order_) {
    CUDA_CALL(cudaMemcpy(data_, a_ptr->data_, getNumDataBytes(), cudaMemcpyHostToDevice));
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
  mexAssert(!a.empty() && !b.empty(), "In DeviceToHost one of the arrays is empty");
  mexAssert(a.size1() == b.size1_ && a.size2() == b.size2_,
    "In DeviceToHost the sizes of matrices do not correspond");  
  // conversion is to get access to protected members  
  const MatGPU *b_ptr = &b;
  MatGPU *a_ptr = static_cast<MatGPU*>(&a);  
  mexAssert(a_ptr->stride_ == 1 && b_ptr->stride_ == 1, "In HostToDevice strides should be 1");
  if (a.order() == b.order()) {
    CUDA_CALL(cudaMemcpy(a_ptr->data_, b_ptr->data_, b_ptr->getNumDataBytes(), cudaMemcpyDeviceToHost));
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
  mexAssert(a_ptr->order_ == true, "In SubSet 'a.order_' should be true");
  mexAssert(b.order_ == false, "In SubSet 'b.order_' should be false");  
  mexAssert(offset + b.size1_ <= a_ptr->size1_ && b.size2_ == a_ptr->size2_,
            "In SubSet the sizes don't correspond each other");  
  MatCPU as;
  as.attach(a_ptr->data_ + offset * a_ptr->size2_, b.size1_, b.size2_, 1, true);
  // we declare additional array here in order to use buffers
  MatGPU br;
  int bufsize = (int) (b.size1_ * b.size2_);
  MatGPU::swapWithBuffer(br, bufsize);
  br.resize(b.size1_, b.size2_);
  br.order_ = true;
  if (dir) {        
    br = as; // HostToDevice    
    b = br; //changes the order, if necessary
  } else {
    br = b; //changes the order, if necessary
    DeviceToHost(br, as);
  }
  MatGPU::swapWithBuffer(br, bufsize);
  
  if (print >= 3) {
    MatGPU::MeasureCudaTime("SubSet");
  }
}

// friend functions

// exact copy from MatCPU
void InitMaps(const MatGPU &a, const std::vector<size_t> &mapsize,
              std::vector< std::vector<MatGPU> > &matrices) {
  mexAssert(kMapsOrder == false, "In InitMaps kMapsOrder should be always false");
  mexAssert(mapsize.size() == 2, "In InitMaps the size vector length != 2");
  // splitting the 2nd dimension
  size_t batchsize = a.size1_, pixels_num = a.size2_;
  if (matrices.size() != batchsize) matrices.resize(batchsize);
  size_t numel = mapsize[0] * mapsize[1];
  mexAssert(pixels_num % numel == 0, "In 'MatGPU::InitMaps' the matrix sizes do not correspond");
  size_t outputmaps = pixels_num / numel;
  for (size_t k = 0; k < batchsize; ++k) {
    if (matrices[k].size() != outputmaps) matrices[k].resize(outputmaps);
    size_t ind = 0;
    for (size_t j = 0; j < outputmaps; ++j) {
      if (!a.order_) {
        matrices[k][j].attach(a.data_ + ind * batchsize + k,
          mapsize[0], mapsize[1], batchsize, kMapsOrder);
      } else {
        matrices[k][j].attach(a.data_ + k * pixels_num + ind,
          mapsize[0], mapsize[1], 1, kMapsOrder);
      }
      ind += numel;
    }    
  }
}

void Sum(MatGPU &a, MatGPU &vect, size_t dim) {  
  if (dim == 1) {
    mexAssert(vect.size1_ == 1 && vect.size2_ == a.size2_,
      "In Sum the sizes do not correspond each other");    
  } else if (dim == 2) {
    mexAssert(vect.size1_ == a.size1_ && vect.size2_ == 1,
      "In Sum the sizes do not correspond each other");       
  } else {
    mexAssert(false, "In MatGPU Sum the dimension parameter must be either 1 or 2");
  }  
  cuda_sumvect(a, vect, dim);  
}

void Mean(MatGPU &a, MatGPU &vect, size_t dim) {
  Sum(a, vect, dim);  
  if (dim == 1) {    
    vect /= (ftype) a.size1_;    
  } else if (dim == 2) {    
    vect /= (ftype) a.size2_;    
  } else {
    mexAssert(false, "In MatGPU Mean the dimension parameter must be either 1 or 2");
  }  
}

void Trans(const MatGPU &a, MatGPU &b) {
  // no resize to ensure that b.data_ is not relocated
  cuda_trans(a, b);  
}

// layer transformation functions

void Prod(const MatGPU &a, bool a_tr, const MatGPU &b, bool b_tr, MatGPU &c) {  
  
  mexAssert(a.order_ == b.order_ && b.order_ == c.order_, "In Prod the orders should be the same");
  mexAssert(a.stride_ == 1 && b.stride_ == 1 && c.stride_ == 1, "In Prod one of strides is not 1"); 
  
  cudaStream_t stream = MatGPU::_defaultStream;
  cublasHandle_t handle = MatGPU::_cublasHandle;
  const ftype scale_prod = 1.0, scale_cur = 0.0;    
  CUBLAS_CALL(cublasSetStream(handle, stream));    
  
  size_t as1, as2, bs1, bs2;
  cublasOperation_t a_op, b_op;    
  
  if (a.order_ == false) { // Alex kernels
    if (!a_tr) { // a
      as1 = a.size1_; as2 = a.size2_;
      a_op = CUBLAS_OP_N;
    } else { // aT
      as1 = a.size2_; as2 = a.size1_;    
      a_op = CUBLAS_OP_T;
    }
    if (!b_tr) { //b
      bs1 = b.size1_; bs2 = b.size2_;
      b_op = CUBLAS_OP_N;
    } else { //bT
      bs1 = b.size2_; bs2 = b.size1_;
      b_op = CUBLAS_OP_T;
    }

    mexAssert(as2 == bs1, "In Prod the sizes of matrices do not correspond");   
    mexAssert(c.size1_ == as1 && c.size2_ == bs2, "In Prod the size of output matrix is wrong"); 
    
    CUBLAS_CALL(cublasSgemm(handle, a_op, b_op, (int) as1, (int) bs2, (int) as2,
                           &scale_prod, a.data_, (int) a.size1_, b.data_, (int) b.size1_,
                           &scale_cur, c.data_, (int) c.size1_));                         
  } else { // cuDNN kernels
    if (!a_tr) { // a
      as1 = a.size2_; as2 = a.size1_;
      a_op = CUBLAS_OP_T;
    } else { // aT
      as1 = a.size1_; as2 = a.size2_;    
      a_op = CUBLAS_OP_N;
    }
    if (!b_tr) { //b
      bs1 = b.size2_; bs2 = b.size1_;
      b_op = CUBLAS_OP_T;
    } else { //bT
      bs1 = b.size1_; bs2 = b.size2_;
      b_op = CUBLAS_OP_N;
    }

    mexAssert(as1 == bs2, "In Prod the sizes of matrices do not correspond");   
    mexAssert(c.size1_ == as2 && c.size2_ == bs1, "In Prod the size of output matrix is wrong"); 
    
    CUBLAS_CALL(cublasSgemm(handle, b_op, a_op, (int) bs1, (int) as2, (int) bs2,
                           &scale_prod, b.data_, (int) b.size2_, a.data_, (int) a.size2_,
                           &scale_cur, c.data_, (int) c.size2_));
  }
}

void TransformActs(const MatGPU &images, MatGPU &targets, 
                   const std::vector<size_t> &prev_mapsize, const std::vector<size_t> &mapsize,
                   const std::vector<ftype> &shift, const std::vector<ftype> &scale, 
                   const std::vector<bool> &mirror, ftype angle, ftype defval) {
      
  size_t numdim = 2;
  mexAssert(shift.size() == numdim && scale.size() == numdim && mirror.size() == numdim,
    "In TransformActs one of the vectors has the wrong size");
  
  size_t batchsize = images.size1();
  MatGPU shift_mat, scale_mat, mirror_mat, angle_mat;  
  
  MatGPU::swapWithBuffer(shift_mat, -3);
  shift_mat.resize(batchsize, numdim);
  shift_mat.assign(0);
  
  MatGPU::swapWithBuffer(scale_mat, -4);
  scale_mat.resize(batchsize, numdim);
  scale_mat.assign(1);
  
  MatGPU::swapWithBuffer(mirror_mat, -5);
  mirror_mat.resize(batchsize, numdim);
  mirror_mat.assign(0);
  
  MatGPU::swapWithBuffer(angle_mat, -6);
  angle_mat.resize(batchsize, 1);
  angle_mat.assign(0);
  
  MatGPU dim_vect;
  //MatGPU randvect(batchsize, 1);
  mexAssert(kDefaultOrder == false, "In TransformActs the default order should be false");  
  for (size_t j = 0; j < numdim; ++j) {    
    if (shift[j] > 0) {
      dim_vect.attach(shift_mat.data_ + batchsize * j, batchsize, 1);
      (dim_vect.rand() *= 2) -= 1;
      dim_vect *= shift[j];
      /*
      (randvect.rand() *= 2) -= 1;
      dim_vect.CondAssign(randvect, false, -0.333, -shift[j]);      
      dim_vect.CondAssign(randvect, true, 0.333, shift[j]);
      */
    }
    if (scale[j] > 1) {
      dim_vect.attach(scale_mat.data_ + batchsize * j, batchsize, 1);
      (dim_vect.rand() *= 2) -= 1;
      (dim_vect *= log(scale[j])).Exp();
      /*
      (randvect.rand() *= 2) -= 1;      
      dim_vect.CondAssign(randvect, false, -0.333, (float) 1 / scale[j]);      
      dim_vect.CondAssign(randvect, true, 0.333, scale[j]);
      */
    }
    if (mirror[j] == true) {
      dim_vect.attach(mirror_mat.data_ + batchsize * j, batchsize, 1);
      dim_vect.rand();    
    }    
  } 
  if (angle > 0) {
    ((angle_mat.rand() *= 2) -= 1) *= kPi * angle;
    /*
    (randvect.rand() *= 2) -= 1;      
    angle_mat.CondAssign(randvect, false, -0.333, -angle);      
    angle_mat.CondAssign(randvect, true, 0.333, angle);
    */
  }
  
  _transformActs(images, targets,
                 prev_mapsize[0], prev_mapsize[1], mapsize[0], mapsize[1], 
                 shift_mat, scale_mat, mirror_mat, angle_mat, defval);  
                 
  MatGPU::swapWithBuffer(shift_mat, -3);
  MatGPU::swapWithBuffer(scale_mat, -4);
  MatGPU::swapWithBuffer(mirror_mat, -5);
  MatGPU::swapWithBuffer(angle_mat, -6);
}

// filter functions

void FilterActs(MatGPU& images, MatGPU& filters, MatGPU& targets,
                const std::vector<size_t> &prev_mapsize,
                size_t filterSize, size_t padding, bool conv) {
  
  if (print >= 3) {
    MatGPU::StartCudaTimer();
  }
  
  #if USE_CUDNN == 0
    _filterActs(images, filters, targets, 
                prev_mapsize[0], prev_mapsize[1], 
                filterSize, padding, conv);
  #else
    
  #endif
  
  if (print >= 3) {
    MatGPU::MeasureCudaTime("FilterActs");
  }
}

void ImgActs(MatGPU& hidActs, MatGPU& filters, MatGPU& targets,
             const std::vector<size_t> &prev_mapsize, 
             size_t filterSize, size_t padding, bool conv) {

  if (print >= 3) {
    MatGPU::StartCudaTimer();
  }
             
  _imgActs(hidActs, filters, targets, 
          prev_mapsize[0], prev_mapsize[1],
          filterSize, padding, conv);  
  
  if (print >= 3) {
    MatGPU::MeasureCudaTime("ImgActs");
  }
  
}

void WeightActs(MatGPU& images, MatGPU& hidActs, MatGPU& targets,
                const std::vector<size_t> &prev_mapsize, 
                size_t filtersize, size_t padding, 
                size_t sum_width, bool conv) {
  size_t mapsize1 = prev_mapsize[0] + 2 * padding + 1 - filtersize;
  size_t mapsize2 = prev_mapsize[1] + 2 * padding + 1 - filtersize;
  size_t chunks_x = DIVUP(mapsize1, sum_width);
  size_t chunks_y = DIVUP(mapsize2, sum_width);
  size_t chunks_num = chunks_x * chunks_y;      
  
  MatGPU tmpbuf;
  int bufsize = (int) (targets.size1_ * targets.size2_ * chunks_num);
  if (conv && chunks_num > 1) {
    MatGPU::swapWithBuffer(tmpbuf, bufsize);    
    tmpbuf.resize(targets.size1_, targets.size2_ * chunks_num);    
  } else {
    tmpbuf.attach(targets);
  }
  
  if (print >= 3) {
    MatGPU::StartCudaTimer();
  }
  
  _weightActs(images, hidActs, tmpbuf,
              prev_mapsize[0], prev_mapsize[1],
              filtersize, padding, chunks_num, sum_width);      
  
  if (print >= 3) {
    MatGPU::MeasureCudaTime("WeightActs");
  }
  
  if (conv && chunks_num > 1) {
    size_t outputmaps = targets.size1_;  
    tmpbuf.reshape(targets.size1_ * targets.size2_, chunks_num);
    targets.reshape(targets.size1_ * targets.size2_, 1);
    Sum(tmpbuf, targets, 2);
    targets.reshape(outputmaps, targets.size1_ / outputmaps);    
    tmpbuf.reshape(targets.size1_, targets.size2_ * chunks_num);
    MatGPU::swapWithBuffer(tmpbuf, bufsize);    
  }
}

// scaling functions
void AvgPooling(MatGPU& images, MatGPU& targets,
                const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride) {
  _convLocalAvgPool(images, targets, prev_mapsize[0], prev_mapsize[1], scale, stride);  
}
void MaxPooling(MatGPU& images, MatGPU& targets,
                const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride) {
  _convLocalMaxPool(images, targets, prev_mapsize[0], prev_mapsize[1], scale, stride);  
}

void AvgPoolingUndo(MatGPU& avgGrads, MatGPU& targets,
                    const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride) {
  _convLocalAvgUndo(avgGrads, targets, prev_mapsize[0], prev_mapsize[1], scale, stride);  
}

void MaxPoolingUndo(MatGPU& images, MatGPU& maxActs, MatGPU& maxGrads, MatGPU& imgGrads,
                    const std::vector<size_t> &prev_mapsize, size_t scale, size_t stride, bool dir) {                    
  if (dir) {
    // imgGrads are targets
    _convLocalMaxUndo(images, maxActs, maxGrads, imgGrads, prev_mapsize[0], prev_mapsize[1], scale, stride);    
  } else {    
    // maxGrads are targets
    _convLocalMaxUndoDer(images, maxActs, imgGrads, maxGrads, prev_mapsize[0], prev_mapsize[1], scale, stride);    
  }  
}

void LocalResponseNorm(MatGPU& images, MatGPU& targets,
                       const std::vector<size_t> &prev_mapsize, size_t normsize, ftype scale, ftype pow) {
  _convContrastNormCrossMap(images, images, targets, prev_mapsize[0], prev_mapsize[1], normsize, scale, pow);
}
void LocalResponseNormUndo(MatGPU& images, MatGPU& maxActs, MatGPU& maxGrads, MatGPU& imgGrads,
                           const std::vector<size_t> &prev_mapsize, size_t normsize, ftype scale, ftype pow) {
  _convResponseNormCrossMapUndo(images, maxActs, maxGrads, imgGrads, prev_mapsize[0], prev_mapsize[1], normsize, scale, pow);
}

// class nonspecific

ftype MatGPU::sum() const {
  return cuda_sum(*this);  
}
