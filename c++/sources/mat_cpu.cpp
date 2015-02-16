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

#include "mat_cpu.h"
#include <algorithm>
#include <cmath>

std::default_random_engine MatCPU::_generator;
  
void MatCPU::InitRand(size_t seed) {
  _generator.seed((unsigned long) seed);
}

/*
MatCPU MatCPU::operator () (size_t ind) {
  MatCPU val_mat;
  val_mat.attach(&data(ind), 1, 1);
  return val_mat;
}*/

// memory functions

MatCPU::MatCPU() {
  //mexPrintMsg("Array constructor 0");
  init();
  //mexPrintMsg("Array constructor 0 end");
}

MatCPU::MatCPU(size_t size1, size_t size2) {
  //mexPrintMsg("Array constructor 1");
  init();
  resize(size1, size2);
  //mexPrintMsg("Array constructor 1 end");
}

MatCPU::MatCPU(const std::vector<size_t> &newsize) {
  //mexPrintMsg("Array constructor 2");
  mexAssert(newsize.size() == 2, "In MatCPU::MatCPU the size vector length != 2");  
  init();
  resize(newsize[0], newsize[1]);
  //mexPrintMsg("Array constructor 2 end");
}

MatCPU::MatCPU(const MatCPU &a) {
  //mexPrintMsg("Array copy constructor");
  init();
  if (a.empty()) return;
  resize(a.size1_, a.size2_); 
  (*this) = a;
  //mexPrintMsg("Array copy constructor end");
}

MatCPU::MatCPU(MatCPU &&a) {
  //mexPrintMsg("Array move constructor");   
  init();
  Swap(*this, a);
  //mexPrintMsg("Array move constructor end");
}

MatCPU::~MatCPU() {
  //mexPrintMsg("Array destructor");
  clear();
  //mexPrintMsg("Array destructor end");
}

MatCPU& MatCPU::init() {
  data_ = NULL;
  size1_ = 0;
  size2_ = 0;  
  stride_ = 1;
  order_ = kDefaultOrder;
  owner_ = false;
  return *this;
}

// only this function supports order_ == true
MatCPU& MatCPU::operator = (const MatCPU &a) {
  //mexPrintMsg("Array const assignment");
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In MatCPU::operator = the arrays have different size");  
  if (order_ == a.order_) {
    for (size_t i = 0; i < size1_ * size2_; ++i) {    
      data(i) = a.data(i);
    }
  } else if (a.order_ != kDefaultOrder) { // order_ == kDefaultOrder    
    MatCPU ar;
    ar.attach(a.data_, a.size2_, a.size1_, a.stride_, kDefaultOrder);    
    Trans(ar, *this);
  } else { // a.order_ == kDefaultOrder, order_ == !kDefaultOrder      
    MatCPU ar;
    ar.attach(data_, size2_, size1_, stride_, kDefaultOrder);    
    Trans(a, ar);
  }
  //mexPrintMsg("Array const assignment end");
  return *this;  
}

MatCPU& MatCPU::resize(size_t size1, size_t size2) {
  //mexPrintMsg("Array resize");
  if (size1 * size2 != size1_ * size2_) {
    clear();
    if (size1 * size2 > 0) {
      data_ = new ftype[size1 * size2];
      owner_ = true;
    }
  }
  size1_ = size1;
  size2_ = size2;
  //mexPrintMsg("Array resize end");  
  return *this;  
}

MatCPU& MatCPU::reshape(size_t size1, size_t size2) {
  mexAssert(size1_ * size2_ == size1 * size2,
    "In MatCPU::reshape the sizes do not correspond");
  size1_ = size1;
  size2_ = size2;
  return *this;
}

MatCPU& MatCPU::reorder(bool order, bool real) {
  //mexAssert(order_ == order, "In MatCPU::reorder orders should be the same");
  if (order_ != order) {
    if (real && size1_ > 1 && size2_ > 1) {
      MatCPU m(size1_, size2_);
      m.order_ = !order_;
      m = (*this);
      order_ = m.order_;
      (*this) = m;  
    } else {
      order_ = !order_;
    }
  }
  return *this;
}

MatCPU& MatCPU::attach(const MatCPU &a) {
  return attach(a.data_, a.size1_, a.size2_, a.stride_, a.order_);
}

MatCPU& MatCPU::attach(const MatCPU &a, size_t offset, size_t size1, size_t size2, bool order) {  
  mexAssert(a.size1_ == 1 || a.size2_ == 1, "In MatCPU::attach with offset one of sizes should be 1");
  mexAssert(offset + size1 * size2 <= a.size1_ * a.size2_, 
            "In MatCPU::attach the sizes don't correspond each other");
  return attach(a.data_ + offset, size1, size2, a.stride_, order);    
}

MatCPU& MatCPU::attach(ftype *ptr, size_t size1, size_t size2) {
  return attach(ptr, size1, size2, 1, kDefaultOrder);
}

MatCPU& MatCPU::attach(ftype *ptr, size_t size1, size_t size2, size_t stride, bool order) {  
  //mexAssert(order == false, "In MatCPU::attach order should be always false");  
  clear();
  data_ = ptr;
  size1_ = size1;
  size2_ = size2;  
  stride_ = stride;    
  order_ = order;
  return *this;
}

MatCPU& MatCPU::clear() {
  //mexPrintMsg("Array clear");
  if (owner_) {
    delete [] data_;
    owner_ = false;
  }
  return init();
  //mexPrintMsg("Array clear end");
}

void Swap(MatCPU &a, MatCPU &b) {
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
}

// data functions

MatCPU& MatCPU::assign(ftype val) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = val;
  }
  return *this;
}

MatCPU& MatCPU::rand() {
  std::uniform_real_distribution<ftype> distribution(0.0, 1.0);
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = distribution(MatCPU::_generator);
  }
  return *this;
}

MatCPU& MatCPU::randnorm() {
  std::normal_distribution<ftype> distribution(0.0, 1.0);
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = distribution(MatCPU::_generator);
  }
  return *this;
}

MatCPU& MatCPU::operator += (const MatCPU &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In MatCPU::+= the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) += a(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator -= (const MatCPU &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In MatCPU::-= the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) -= a(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator *= (const MatCPU &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_, 
    "In 'MatCPU::*=' the matrices are of the different size");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) *= a(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator /= (const MatCPU &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_, 
    "In 'MatCPU::/=' the matrices are of the different size");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) /= a(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator += (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) += a;
  }
  return *this;
}

MatCPU& MatCPU::operator -= (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) -= a;
  }
  return *this;
}

MatCPU& MatCPU::operator *= (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) *= a;
  }
  return *this;
}

MatCPU& MatCPU::operator /= (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) /= a;
  }
  return *this;
}

MatCPU& MatCPU::Sign() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    if (data(i) > kEps) {
      data(i) = 1;
    } else if (data(i) < -kEps) {
      data(i) = -1;
    } else {
      data(i) = 0;
    }
  }
  return *this;
}

MatCPU& MatCPU::Sqrt() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = (ftype) sqrt((double) data(i));
  }
  return *this;
}

MatCPU& MatCPU::Exp() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = exp(data(i));
  }
  return *this;
}

MatCPU& MatCPU::Log() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = log(data(i));
  }
  return *this;
}

MatCPU& MatCPU::SoftMax() {

  for (size_t i = 0; i < size1_; ++i) {
    ftype max_val = data(i, 0);
    for (size_t j = 1; j < size2_; ++j) {
      if (data(i, j) > max_val) max_val = data(i, j);
    }
    ftype sum_exp = 0;
    for (size_t j = 0; j < size2_; ++j) {
      sum_exp += exp(data(i, j) - max_val);
    }
    ftype log_sum_exp = log(sum_exp) + max_val;
    
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) = exp(data(i, j) - log_sum_exp);
    }
  }
  return *this;
}

MatCPU& MatCPU::SoftDer(const MatCPU& a) {
  for (size_t i = 0; i < size1_; ++i) {
    ftype comsum = 0;
    for (size_t j = 0; j < size2_; ++j) {
      comsum += data(i, j) * a(i, j);
    }
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) = a(i, j) * (data(i, j) - comsum);
    }
  }
  return *this;
}

MatCPU& MatCPU::Sigmoid() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data(i) = (1 + tanh(data(i)/2)) / 2;
    //data(i) = 1 / (1 + exp(-data(i)));
  }
  return *this;
}

MatCPU& MatCPU::SigmDer(const MatCPU& a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_, 
    "In 'MatCPU::SigmDer' the matrices are of the different size");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) *= a(i, j) * (1 - a(i, j));      
    }
  }  
  return *this;
}  

MatCPU& MatCPU::CondAssign(const MatCPU &condmat, bool incase, ftype threshold, ftype a) {
  mexAssert(size1_ == condmat.size1_ && size2_ == condmat.size2_,
    "In MatCPU::CondAssign the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      if (incase == (condmat(i, j) > threshold)) data(i, j) = a; // xor      
    }
  }
  return *this;
}

MatCPU& MatCPU::CondAdd(const MatCPU &condmat, bool incase, ftype threshold, ftype a) {
  mexAssert(size1_ == condmat.size1_ && size2_ == condmat.size2_,
    "In MatCPU::CondAdd the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      if (incase == (condmat(i, j) > threshold)) data(i, j) += a; // xor      
    }
  }
  return *this;
}

MatCPU& MatCPU::CondMult(const MatCPU &condmat, bool incase, ftype threshold, ftype a) {
  mexAssert(size1_ == condmat.size1_ && size2_ == condmat.size2_,
    "In MatCPU::CondMult the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      if (incase == (condmat(i, j) > threshold)) data(i, j) *= a; // xor      
    }
  }
  return *this;
}

MatCPU& MatCPU::AddVect(const MatCPU &vect, size_t dim) {  

  if (dim == 1) {
    mexAssert(vect.size1_ == 1, "In 'MatCPU::AddVect' the first dimension must be 1"); 
    mexAssert(size2_ == vect.size2_,
      "In 'MatCPU::AddVect' the second dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) += vect(0, j);
      }
    }
  } else if (dim == 2) {
    mexAssert(vect.size2_ == 1, "In 'MatCPU::AddVect' the second dimension must be 1"); 
    mexAssert(size1_ == vect.size1_,
      "In 'MatCPU::AddVect' the first dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) += vect(i, 0);
      }
    }
  } else {
    mexAssert(false, "In MatCPU::AddVect the dimension parameter must be either 1 or 2");
  }
  return *this;
}

MatCPU& MatCPU::MultVect(const MatCPU &vect, size_t dim) {  
  if (dim == 1) {
    mexAssert(vect.size1_ == 1, "In 'MatCPU::MultVect' the first dimension must be 1"); 
    mexAssert(size2_ == vect.size2_,
      "In 'MatCPU::MultVect' the second dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) *= vect(0, j);
      }
    }
  } else if (dim == 2) {
    mexAssert(vect.size2_ == 1, "In 'MatCPU::MultVect' the second dimension must be 1"); 
    mexAssert(size1_ == vect.size1_,
      "In 'MatCPU::MultVect' the first dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) *= vect(i, 0);
      }
    }
  } else {
    mexAssert(false, "In MatCPU::MultVect the dimension parameter must be either 1 or 2");
  }
  return *this;
}

MatCPU& MatCPU::Normalize(ftype norm) {
  mexAssert(order_ == true, "In Normalize the order_ should be true");
  for (size_t i = 0; i < size1_; ++i) {
    ftype curnorm = 0;
    ftype *dptr = data_ + i * size2_;
    for (size_t j = 0; j < size2_; ++j) {
      curnorm += dptr[j] * dptr[j];
    }
    curnorm = (ftype) sqrt((double) curnorm);
    if (curnorm > kEps) {
      for (size_t j = 0; j < size2_; ++j) {
        dptr[j] *= (norm / curnorm);
      }
    }
  }
  return *this;
}

MatCPU& MatCPU::Validate() {  
  for (size_t i = 0; i < size1_ * size2_; ++i) {  
    if (-kEps < data(i) && data(i) < kEps) data(i) = 0;
  }
  return *this;
}

// friend functions

void Sum(const MatCPU &a, MatCPU &vect, size_t dim) {
   
  if (dim == 1) {    
    mexAssert(vect.size1_ == 1 && vect.size2_ == a.size2_,
      "In Sum the sizes do not correspond each other");    
    vect.assign(0);
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < a.size2_; ++j) {
        vect(0, j) += a(i, j);        
      }
    }    
  } else if (dim == 2) {    
    mexAssert(vect.size1_ == a.size1_ && vect.size2_ == 1,
      "In Sum the sizes do not correspond each other");    
    vect.assign(0);
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < a.size2_; ++j) {
        vect(i, 0) += a(i, j);        
      }     
    }    
  } else {
    mexAssert(false, "In MatCPU Sum the dimension parameter must be either 1 or 2");
  }  
}

void Mean(const MatCPU &a, MatCPU &vect, size_t dim) {  
  Sum(a, vect, dim);  
  if (dim == 1) {
    vect /= (ftype) a.size1_;
  } else if (dim == 2) {    
    vect /= (ftype) a.size2_;
  } else {
    mexAssert(false, "In MatCPU Mean the dimension parameter must be either 1 or 2");
  }
}

void Trans(const MatCPU &a, MatCPU &b) {
  // no resize to ensure that b.data_ is not relocated
  mexAssert(a.size1_ == b.size2_ && a.size2_ == b.size1_,
            "In Trans the sizes of matrices do not correspond");  
  mexAssert(a.data_ != b.data_, "In Trans the matrices are the same");  
  for (size_t i = 0; i < b.size1_; ++i) { 
    for (size_t j = 0; j < b.size2_; ++j) { 
      b(i, j) = a(j, i);
    }
  }
}

/*
 * a - main matrix, b - submatrix
 * dir == true  means a -> b,
 * dir == false means a <- b
 */
void SubSet(MatCPU &a, MatCPU &b, size_t offset, bool dir) {  

  mexAssert(a.order_ == true, "In SubSet 'a.order_' should be true");
  mexAssert(b.order_ == false, "In SubSet 'b.order_' should be false");  
  mexAssert(offset + b.size1_ <= a.size1_ && b.size2_ == a.size2_,
            "In SubSet the sizes don't correspond each other");  
  MatCPU as;
  as.attach(a.data_ + offset * a.size2_, b.size1_, b.size2_, 1, true);
  if (dir) {    
    b = as;
  } else {
    as = b;
  }
}

void Shuffle(MatCPU &a, MatCPU &b) {
  mexAssert(a.order_ == true && b.order_ == true, "In Shuffle the orders should be true");
  mexAssert(a.stride_ == 1 && b.stride_ == 1, "In Shuffle the strides should be 1");
  mexAssert(a.size1_ == b.size1_, "In Shuffle the sizes do not correspond");
  size_t train_num = a.size1_;
  //mexPrintMsg("train_num", train_num);
  std::vector<size_t> randind(train_num);
  for (size_t i = 0; i < train_num; ++i) {
    randind[i] = i;
  }
  std::random_shuffle(randind.begin(), randind.end());    
  
  MatCPU a_new(a.size1_, a.size2_);
  MatCPU b_new(b.size1_, b.size2_);  
  for (size_t i = 0; i < train_num; ++i) {
    ftype *a_ptr = a.data_ + i * a.size2_;
    ftype *a_new_ptr = a_new.data_ + randind[i] * a.size2_;
    ftype *b_ptr = b.data_ + i * b.size2_;
    ftype *b_new_ptr = b_new.data_ + randind[i] * b.size2_;
    for (size_t j = 0; j < a.size2_; ++j) {
      a_new_ptr[j] = a_ptr[j];      
    }
    for (size_t j = 0; j < b.size2_; ++j) {
      b_new_ptr[j] = b_ptr[j];
    }
  }
  a_new.order_ = true;
  b_new.order_ = true;
  
  Swap(a, a_new);
  Swap(b, b_new);  
}

void InitMaps(const MatCPU &a, const std::vector<size_t> &mapsize,
              std::vector< std::vector<MatCPU> > &matrices) {
  mexAssert(kMapsOrder == false, "In InitMaps kMapsOrder should be always false");
  mexAssert(mapsize.size() == 2, "In InitMaps the size vector length != 2");
  // splitting the 2nd dimension
  size_t batchsize = a.size1_, pixels_num = a.size2_;
  if (matrices.size() != batchsize) matrices.resize(batchsize);
  size_t numel = mapsize[0] * mapsize[1];
  mexAssert(pixels_num % numel == 0, "In 'MatCPU::InitMaps' the matrix sizes do not correspond");
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

// layer transformation functions

void Prod(const MatCPU &a, bool a_tr, const MatCPU &b, bool b_tr, MatCPU &c) {
  size_t as1, as2, bs1, bs2;
  MatCPU al, bl, cl;
  if (!a_tr) { // a
    as1 = a.size1_; as2 = a.size2_;    
    al.resize(a.size1_, a.size2_);
    al = a;    
  } else { // aT
    as1 = a.size2_; as2 = a.size1_;       
    al.resize(a.size2_, a.size1_);
    Trans(a, al); 
  }
  if (!b_tr) { //b
    bs1 = b.size1_; bs2 = b.size2_;    
    bl.resize(b.size2_, b.size1_);
    Trans(b, bl); 
  } else { //bT
    bs1 = b.size2_; bs2 = b.size1_;    
    bl.resize(b.size1_, b.size2_);
    bl = b;    
  }
  mexAssert(as2 == bs1, "In Prod the sizes of matrices do not correspond"); 
  mexAssert(c.size1_ == as1 && c.size2_ == bs2, "In Prod the size of output matrix is wrong");
  
  cl.resize(as1, bs2);
  cl.assign(0);
  for (size_t k = 0; k < bs1; ++k) {    
    for (size_t j = 0; j < bs2; ++j) {
      for (size_t i = 0; i < as1; ++i) {                    
        cl(i, j) += al(i, k) * bl(j, k);            
      }
    }
  }  
  c = cl;
  //Swap(c, cl);  
}

void Filter(const MatCPU &image, const MatCPU &filter, 
            const std::vector<size_t> padding, bool conv, MatCPU &filtered) {
  mexAssert(filtered.size1_ == image.size1_ + 2*padding[0] + 1 - filter.size1_ &&   
            filtered.size2_ == image.size2_ + 2*padding[1] + 1 - filter.size2_,
            "In 'Filter' the parameters do not correspond each other");
  if (padding[0] == 0 && padding[1] == 0) {
    size_t fsize1 = filter.size1_;
    size_t fsize2 = filter.size2_;        
    for (size_t i = 0; i < filtered.size1_; ++i) {
      for (size_t j = 0; j < filtered.size2_; ++j) {
        ftype val = 0;
        if (!conv) {
          for (size_t u = 0; u < fsize1; ++u) {
            for (size_t v = 0; v < fsize2; ++v) {
              val += filter(u, v) * image(i + u, j + v);
            }        
          }
        } else {
          for (size_t u = 0; u < fsize1; ++u) {
            for (size_t v = 0; v < fsize2; ++v) {
              val += filter(fsize1 - 1 - u, fsize2 - 1 - v) * image(i + u, j + v);
            }        
          }
        }
        filtered(i, j) = val;
      }
    }
  } else {
    int pad1 = (int) padding[0];
    int pad2 = (int) padding[1];
    int fsize1 = (int) filter.size1_;
    int fsize2 = (int) filter.size2_;
    int offs1 = (int) filtered.size1_ - 1 + fsize1 - pad1;
    int offs2 = (int) filtered.size2_ - 1 + fsize2 - pad2;        
    for (int i = 0; i < filtered.size1_; ++i) {
      for (int j = 0; j < filtered.size2_; ++j) {
        int minu = MAX(pad1 - i, 0);
        int minv = MAX(pad2 - j, 0);
        int maxu = MIN(offs1 - i, fsize1);
        int maxv = MIN(offs2 - j, fsize2);
        ftype val = 0;
        if (!conv) {
          for (int u = minu; u < maxu; ++u) {
            for (int v = minv; v < maxv; ++v) {
              val += filter(u, v) * image(i + u - pad1, j + v - pad2);
            }        
          }
        } else {
          for (int u = minu; u < maxu; ++u) {
            for (int v = minv; v < maxv; ++v) {
              val += filter(fsize1 - 1 - u, fsize2 - 1 - v) * image(i + u - pad1, j + v - pad2);
            }        
          }
        }
        filtered(i, j) = val;        
      }
    }
  }  
}

void Transform(const MatCPU &image, const std::vector<ftype> &shift,
               const std::vector<ftype> &scale, const std::vector<bool> &mirror, 
               ftype angle, ftype defval, MatCPU &transformed) {
  ftype m1 = (ftype) image.size1_ / 2 - (ftype) 0.5;
  ftype m2 = (ftype) image.size2_ / 2 - (ftype) 0.5;
  ftype n1 = (ftype) transformed.size1_ / 2 - (ftype) 0.5;
  ftype n2 = (ftype) transformed.size2_ / 2 - (ftype) 0.5;
  ftype angcos = cos(angle);
  ftype angsin = sin(angle);
  for (size_t i = 0; i < transformed.size1_; ++i) {
    for (size_t j = 0; j < transformed.size2_; ++j) {
      ftype xi1 = (i - n1) * scale[0];
      ftype xi2 = (j - n2) * scale[1];
      ftype x1 = xi1 * angcos - xi2 * angsin + m1 + shift[0];
      ftype x2 = xi1 * angsin + xi2 * angcos + m2 + shift[1];
      if (mirror[0]) x1 = image.size1_ - 1 - x1;
      if (mirror[1]) x2 = image.size2_ - 1 - x2;
      //mexAssert(0 <= x1 && x1 <= image.size1_-2, "x1 is out of range");
      //mexAssert(0 <= x2 && x2 <= image.size2_-2, "x2 is out of range");      
      if (0 <= x1 && x1 <= image.size1_ - 1 &&
          0 <= x2 && x2 <= image.size2_ - 1) {
        size_t xu1 = (size_t) x1;
        size_t xu2 = (size_t) x2;
        size_t xp1 = MIN(xu1 + 1, image.size1_ - 1);
        size_t xp2 = MIN(xu2 + 1, image.size2_ - 1);
        ftype vl = (x1 - (ftype) xu1) * image(xp1, xu2) + ((ftype) xu1 + 1 - x1) * image(xu1, xu2);
        ftype vh = (x1 - (ftype) xu1) * image(xp1, xp2) + ((ftype) xu1 + 1 - x1) * image(xu1, xp2);
        transformed(i, j) = (x2 - (ftype) xu2) * vh + ((ftype) xu2 + 1 - x2) * vl;
      } else {
        transformed(i, j) = defval;
      } 
    }
  }
}

void MeanScale(const MatCPU &image, const std::vector<size_t> &scale,
               const std::vector<size_t> &stride, MatCPU &scaled) {
  mexAssert(scaled.size1_ == DIVUP(image.size1_, stride[0]) &&
            scaled.size2_ == DIVUP(image.size2_, stride[1]),
            "In 'MeanScale' the parameters do not correspond each other");
  scaled.assign(0);
  for (size_t i = 0; i < scaled.size1_; ++i) {
    for (size_t j = 0; j < scaled.size2_; ++j) {
      size_t maxu = MIN(image.size1_ - i*stride[0], scale[0]);
      size_t maxv = MIN(image.size2_ - j*stride[1], scale[1]);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          scaled(i, j) += image(i*stride[0]+u, j*stride[1]+v);
        }
      }
      scaled(i, j) /= (maxu * maxv);
    }
  }  
}

void MeanScaleDer(const MatCPU &scaled_der, const std::vector<size_t> &scale,
                  const std::vector<size_t> &stride, MatCPU &image_der) {
  mexAssert(scaled_der.size1_ == DIVUP(image_der.size1_, stride[0]) && 
            scaled_der.size2_ == DIVUP(image_der.size2_, stride[1]),
            "In 'MeanScaleDer' the parameters do not correspond each other");  
  image_der.assign(0);
  for (size_t i = 0; i < scaled_der.size1_; ++i) {
    for (size_t j = 0; j < scaled_der.size2_; ++j) {
      size_t maxu = MIN(image_der.size1_ - i*stride[0], scale[0]);
      size_t maxv = MIN(image_der.size2_ - j*stride[1], scale[1]);
      ftype scaled_val = scaled_der(i, j) / (maxu * maxv);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          image_der(i*stride[0]+u, j*stride[1]+v) += scaled_val;
        }
      }      
    }
  }
}

void MaxScale(const MatCPU &image, const std::vector<size_t> &scale,
              const std::vector<size_t> &stride, MatCPU &scaled) {
  mexAssert(scaled.size1_ == DIVUP(image.size1_, stride[0]) &&
            scaled.size2_ == DIVUP(image.size2_, stride[1]),
            "In 'MaxScale' the parameters do not correspond each other");
  for (size_t i = 0; i < scaled.size1_; ++i) {
    for (size_t j = 0; j < scaled.size2_; ++j) {
      size_t maxu = MIN(image.size1_ - i*stride[0], scale[0]);
      size_t maxv = MIN(image.size2_ - j*stride[1], scale[1]);
      ftype maxval = image(i*stride[0], j*stride[1]) - 1;      
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          if (maxval < image(i*stride[0]+u, j*stride[1]+v)) {
            maxval = image(i*stride[0]+u, j*stride[1]+v);
          }
        }
      }
      scaled(i, j) = maxval;
    }
  }  
}

void MaxScaleDer(const MatCPU &image, const MatCPU &scaled, MatCPU &scaled_der, MatCPU &image_der, 
                 const std::vector<size_t> &scale, const std::vector<size_t> &stride, bool dir) {
  mexAssert(scaled.size1_ == DIVUP(image.size1_, stride[0]) &&
            scaled.size2_ == DIVUP(image.size2_, stride[1]),
            "In 'MaxScale' the parameters do not correspond each other 1");
  mexAssert(scaled.size1_ == scaled_der.size1_ && scaled.size2_ == scaled_der.size2_,
            "In 'MaxScale' the parameters do not correspond each other 2");
  mexAssert(image.size1_ == image_der.size1_ && image.size2_ == image_der.size2_,
            "In 'MaxScale' the parameters do not correspond each other 3");
  if (dir) {          
    image_der.assign(0);
  } else {
    scaled_der.assign(0);
  }
  for (size_t i = 0; i < scaled.size1_; ++i) {
    for (size_t j = 0; j < scaled.size2_; ++j) {
      size_t maxu = MIN(image.size1_ - i*stride[0], scale[0]);
      size_t maxv = MIN(image.size2_ - j*stride[1], scale[1]);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          if (image(i*stride[0]+u, j*stride[1]+v) == scaled(i, j)) {
            if (dir) {
              image_der(i*stride[0]+u, j*stride[1]+v) += scaled_der(i, j);
            } else {
              scaled_der(i, j) += image_der(i*stride[0]+u, j*stride[1]+v);
            }
          }
        }
      }      
    }
  }  
}

ftype MatCPU::sum() const {
  ftype matsum = 0;
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    matsum += data(i);
  }      
  return matsum;  
}

bool MatCPU::hasZeros() const {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    if (data(i) == 0) return true;
  }
  return false;
}

void MatCPU::write(ftype *vect) const {
  size_t ind = 0;  
  for (size_t i = 0; i < size1_ * size2_; ++i) {      
      vect[ind++] = data(i);      
  } 
}