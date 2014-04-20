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

#include "mat.h"

Mat::Mat() {
  //mexPrintMsg("Array constructor");
  data_ = NULL;
  size1_ = 0;
  size2_ = 0;
  owner_ = false;
  //mexPrintMsg("Array constructor end");
}

Mat::Mat(const std::vector<size_t> &newsize) {
  //mexPrintMsg("Array constructor 2");
  mexAssert(newsize.size() == 2, "In Mat::Mat the size vector length != 2");  
  data_ = new ftype[newsize[0] * newsize[1]];
  size1_ = newsize[0];
  size2_ = newsize[1];
  owner_ = true;
  //mexPrintMsg("Array constructor 2 end");
}

Mat::Mat(size_t size1, size_t size2) {
  //mexPrintMsg("Array constructor 2");
  data_ = new ftype[size1 * size2];
  size1_ = size1;
  size2_ = size2;
  owner_ = true;
  //mexPrintMsg("Array constructor 2 end");
}

Mat::Mat(const Mat &a) {
  //mexPrintMsg("Array copy constructor");
  owner_ = false;
  clear();
  if (a.data_ == NULL) return;
  //mexPrintMsg("Array copy resize");
  resize(a.size1_, a.size2_);  
  for (size_t i = 0; i < size1_ * size2_ ; ++i) { 
    data_[i] = a.data_[i];
  }
  //mexPrintMsg("Array copy constructor end");
}

Mat::Mat(Mat &&a) {
  //mexPrintMsg("Array move constructor");   
  data_ = a.data_;
  size1_ = a.size1_;
  size2_ = a.size2_;
  owner_ = a.owner_;
  a.data_ = NULL;
  a.size1_ = 0;
  a.size2_ = 0;
  a.owner_ = false;
  //mexPrintMsg("Array move constructor end");
}

Mat::~Mat() {
  //mexPrintMsg("Array destructor");
  if (owner_) delete [] data_;  
  //mexPrintMsg("Array destructor end");
}

Mat& Mat::operator = (const Mat &a) {
  //mexPrintMsg("Array const assignment");
  Mat tmp(a);  
  Swap(*this, tmp);  
  //mexPrintMsg("Array const assignment end");
  return *this;  
}

Mat& Mat::operator = (Mat &&a) {  
  //mexPrintMsg("Array move assignment");
  Swap(*this, a);
  //mexPrintMsg("Array move assignment end");
  return *this;
}

void Mat::clear() {
  //mexPrintMsg("Array clear");
  if (owner_) {
    delete [] data_;    
    owner_ = false;
  }
  data_ = NULL;
  size1_ = 0;
  size2_ = 0;  
  //mexPrintMsg("Array clear end");
}

Mat& Mat::attach(ftype *vect, const std::vector<size_t> &newsize) {
  mexAssert(newsize.size() == 2, "In Mat::attach the size vector length != 2");
  return attach(vect, newsize[0], newsize[1]);  
}

Mat& Mat::attach(const Mat &a, const std::vector<size_t> &newsize, size_t offset) {
  mexAssert(newsize.size() == 2, "In Mat::attach the size vector length != 2");
  mexAssert(newsize[0] * newsize[1] + offset <= a.size1_ * a.size2_, 
    "In Mat::attach the sizes are incorrect");
  return attach(a.data_ + offset, newsize[0], newsize[1]);  
}

Mat& Mat::attach(ftype *vect, size_t size1, size_t size2) {
  clear();
  data_ = vect;
  size1_ = size1;
  size2_ = size2;
  return *this;
}

Mat& Mat::attach(const Mat &a) {
  clear();
  data_ = a.data_;
  size1_ = a.size1_;
  size2_ = a.size2_;
  return *this;
}

Mat& Mat::assign(ftype val) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] = val;
  }
  return *this;
}

Mat& Mat::rand() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] = (ftype) std::rand() / RAND_MAX;
  }
  return *this;
}

void Mat::resize(const std::vector<size_t> &newsize) {
  mexAssert(newsize.size() == 2, "In Mat::resize the size vector length != 2");
  resize(newsize[0], newsize[1]);  
}

void Mat::resize(size_t size1, size_t size2) {
  //mexPrintMsg("Array resize");
  if (size1 * size2 != size1_ * size2_) {
    if (owner_) delete [] data_;
    data_ = new ftype[size1 * size2];
    owner_ = true;
  }
  size1_ = size1;
  size2_ = size2;  
  //mexPrintMsg("Array resize end");  
}

Mat& Mat::reshape(size_t size1, size_t size2) {
  mexAssert(size1_ * size2_ == size1 * size2,
    "In Mat::reshape the sizes do not correspond");
  size1_ = size1;
  size2_ = size2;
  return *this;
}

Mat& Mat::copy(const Mat &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In Mat::copy the sizes of matrices must coincide");
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] = a.data_[i];
  }
  return *this;
}

Mat& Mat::init(const std::vector<size_t> &newsize, ftype val) {
  mexAssert(newsize.size() == 2, "In Mat::init the size vector length != 2");
  return init(newsize[0], newsize[1], val);
}

Mat& Mat::init(size_t size1, size_t size2, ftype val) {
  resize(size1, size2);
  assign(val);  
  return *this;
}

void Mat::ToVect(ftype *vect) const {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    vect[i] = data_[i];
  }
}

ftype Mat::Sum() const {
  ftype matsum = 0;
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    matsum += data_[i];
  }      
  return matsum;  
}

ftype& Mat::operator () (size_t ind) {
  mexAssert(size1_ == 1 || size2_ == 1, "In 'Mat::(ind)' matrix is not really a vector");  
  return data_[ind];
}

const ftype& Mat::operator () (size_t ind) const {
  mexAssert(size1_ == 1 || size2_ == 1, "In 'Mat::(ind)' matrix is not really a vector");  
  return data_[ind];
}

Mat& Mat::operator += (const Mat &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In Mat::+= the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] += a.data_[i];
  }
  return *this;
}

Mat& Mat::operator -= (const Mat &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_,
    "In Mat::-= the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] -= a.data_[i];
  }
  return *this;
}

Mat& Mat::operator *= (const Mat &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_, 
    "In 'Mat::*=' the matrices are of the different size");
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] *= a.data_[i];
  }
  return *this;
}

Mat& Mat::operator /= (const Mat &a) {
  mexAssert(size1_ == a.size1_ && size2_ == a.size2_, 
    "In 'Mat::/=' the matrices are of the different size");
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] /= a.data_[i];
  }
  return *this;
}

Mat& Mat::operator += (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] += a;
  }
  return *this;
}

Mat& Mat::operator -= (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] -= a;
  }
  return *this;
}

Mat& Mat::operator *= (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] *= a;
  }
  return *this;
}

Mat& Mat::operator /= (ftype a) {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] /= a;
  }
  return *this;
}

Mat& Mat::Sign() {
  for (size_t i = 0; i < size1_ * size2_ ; ++i) {
    if (data_[i] > kEps) {
      data_[i] = 1;
    } else if (data_[i] < -kEps) {
      data_[i] = -1;
    } else {
      data_[i] = 0;
    }
  }
  return *this;
}

Mat& Mat::SoftMax() {

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

Mat& Mat::SoftDer(const Mat& a) {
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

Mat& Mat::Sigmoid() {
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] = (1 + tanh(data_[i]/2)) / 2;
    //data_[i] = 1 / (1 + exp(-data_[i]));
  }
  return *this;
}

Mat& Mat::SigmDer(const Mat& a) {
  mexAssert(size1_ == a.size1() && size2_ == a.size2(), 
    "In 'Mat::SigmDer' the matrices are of the different size");
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    data_[i] *= a.data_[i] * (1 - a.data_[i]);
  }
  return *this;
}  

Mat& Mat::ElemMax(ftype a) {
  for (size_t i = 0; i < size1_ * size2_ ; ++i) {  
    if (data_[i] < a) data_[i] = a;
  }
  return *this;
}

Mat& Mat::CondAssign(const Mat &condmat, ftype threshold, bool incase, ftype a) {
  mexAssert(size1_ == condmat.size1_ && size2_ == condmat.size2_,
    "In Mat::CondAssign the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_ * size2_ ; ++i) {
    if (incase == (condmat.data_[i] > threshold)) data_[i] = a; // xor
  }
  return *this;
}

Mat& Mat::CondAdd(const Mat &condmat, ftype threshold, bool incase, ftype a) {
  mexAssert(size1_ == condmat.size1_ && size2_ == condmat.size2_,
    "In Mat::CondAdd the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_ * size2_ ; ++i) {
    if (incase == (condmat.data_[i] > threshold)) data_[i] += a; // xor
  }
  return *this;
}

Mat& Mat::CondProd(const Mat &condmat, ftype threshold, bool incase, ftype a) {
  mexAssert(size1_ == condmat.size1_ && size2_ == condmat.size2_,
    "In Mat::CondProd the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_ * size2_ ; ++i) {
    if (incase == (condmat.data_[i] > threshold)) data_[i] *= a; // xor
  }
  return *this;
}

Mat& Mat::AddVect(const Mat &vect, size_t dim) {  
  if (dim == 1) {
    mexAssert(vect.size2_ == 1, "In 'Mat::AddVect' the second dimension must be 1"); 
    mexAssert(size1_ == vect.size1_,
      "In 'Mat::AddVect' the second dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) += vect(i, 0);
      }
    }   
  } else if (dim == 2) {
    mexAssert(vect.size1_ == 1, "In 'Mat::AddVect' the first dimension must be 1"); 
    mexAssert(size2_ == vect.size2_,
      "In 'Mat::AddVect' the first dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) += vect(0, j);
      }
    }
  } else {
    mexAssert(false, "In Mat::AddVect the dimension parameter must be either 1 or 2");
  }
  return *this;
}

Mat& Mat::MultVect(const Mat &vect, size_t dim) {  
  if (dim == 1) {
    mexAssert(vect.size2_ == 1, "In 'Mat::AddVect' the second dimension must be 1"); 
    mexAssert(size1_ == vect.size1_,
      "In 'Mat::AddVect' the second dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) *= vect(i, 0);
      }
    }   
  } else if (dim == 2) {
    mexAssert(vect.size1_ == 1, "In 'Mat::AddVect' the first dimension must be 1"); 
    mexAssert(size2_ == vect.size2_,
      "In 'Mat::AddVect' the first dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < size1_; ++i) {
      for (size_t j = 0; j < size2_; ++j) {
        data(i, j) *= vect(0, j);
      }
    }
  } else {
    mexAssert(false, "In Mat::AddVect the dimension parameter must be either 1 or 2");
  }
  return *this;
}

Mat& Mat::Normalize(ftype norm, ftype &oldnorm) {
  (*this) -= Sum() / (size1_ * size2_);
  ftype curnorm = 0;
  for (size_t i = 0; i < size1_ * size2_; ++i) {
    curnorm += data_[i] * data_[i];
  }
  curnorm = sqrt(curnorm);
  if (curnorm > kEps) {
    (*this) *= (norm / curnorm);
  }
  oldnorm = curnorm;
  return *this;
}

Mat& Mat::CalcPercents() {
  for (size_t i = 0; i < size1_; ++i) {
    ftype maxval = data(i, 0);
    for (size_t j = 1; j < size2_; ++j) {
      if (maxval < data(i, j)) maxval = data(i, j);
    }
    ftype lcs = 0;
    for (size_t j = 0; j < size2_; ++j) {      
      lcs += exp(data(i, j) - maxval);
    }
    lcs = log(lcs) + maxval;
    for (size_t j = 0; j < size2_; ++j) {      
      data(i, j) = exp(data(i, j) - lcs);
    }
  }
  return *this;
}

Mat& Mat::Validate() {  
  for (size_t i = 0; i < size1_ * size2_ ; ++i) {  
    if (-kEps < data_[i] && data_[i] < kEps) data_[i] = 0;
  }
  return *this;
}

// friend functions

void Swap(Mat &a, Mat &b) {
  ftype *data_tmp = b.data_;
  b.data_ = a.data_;
  a.data_ = data_tmp;
  
  size_t size1_tmp = b.size1_;
  b.size1_ = a.size1_;
  a.size1_ = size1_tmp;
  
  size_t size2_tmp = b.size2_;
  b.size2_ = a.size2_;
  a.size2_ = size2_tmp; 
  
  bool owner_tmp = b.owner_;
  b.owner_ = a.owner_;
  a.owner_ = owner_tmp;  
}

Mat Sum(const Mat &a, size_t dim) {
  
  Mat vect;
  if (dim == 1) {    
    vect.init(1, a.size2_, 0);
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < a.size2_; ++j) {
        vect(0, j) += a(i, j);        
      }
    }    
  } else if (dim == 2) {    
    vect.init(a.size1_, 1, 0);
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < a.size2_; ++j) {
        vect(i, 0) += a(i, j);        
      }     
    }    
  } else {
    mexAssert(false, "In Mat Sum(a, dim) the dimension parameter must be either 1 or 2");
  }
  return vect;
}

Mat Mean(const Mat &a, size_t dim) {  
  Mat vect = Sum(a, dim);  
  if (dim == 1) {
    vect /= a.size1_;    
  } else if (dim == 2) {    
    vect /= a.size2_;        
  } else {
    mexAssert(false, "In Mat Mean(a, dim) the dimension parameter must be either 1 or 2");
  }
  return vect;
}

Mat Trans(const Mat &a) {
  Mat c(a.size2_, a.size1_);
  for (size_t i = 0; i < c.size1_; ++i) { 
    for (size_t j = 0; j < c.size2_; ++j) { 
      c(i, j) = a(j, i);
    }
  }
  return c;
}

Mat SubMat(const Mat &a, const std::vector<size_t> &ind, size_t dim) {
  
  mexAssert(ind.size() > 0, "In SubMat the index vector is empty");
  size_t minind = *(std::min_element(ind.begin(), ind.end()));
  mexAssert(minind >= 0, "In SubMat one of the indices is less than zero");  
  Mat submat;
  if (dim == 1) {
    size_t maxind = *(std::max_element(ind.begin(), ind.end()));    
    mexAssert(maxind < a.size1_, "In SubMat one of the indices is larger than the array size");    
    submat.resize(ind.size(), a.size2_);
    for (size_t i = 0; i < ind.size(); ++i) {
      for (size_t j = 0; j < a.size2(); ++j) {
        submat(i, j) = a(ind[i], j);
      }
    }
  } else if (dim == 2) {
    size_t maxind = *(std::max_element(ind.begin(), ind.end()));    
    mexAssert(maxind < a.size2_, "In SubMat one of the indices is larger than the array size");
    submat.resize(a.size1_, ind.size());
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < ind.size(); ++j) {
        submat(i, j) = a(i, ind[j]);
      }
    }    
  } else {
    mexAssert(false, "In Mat::SubMat the second parameter must be either 1 or 2");
  }
  return submat;
}

void InitMaps(const Mat &a, const std::vector<size_t> &mapsize,
              std::vector< std::vector<Mat> > &matrices) {
  mexAssert(mapsize.size() == 2, "In Mat::InitMaps the size vector length != 2");
#if MEM_ORDER == HORIZONTAL
  // splitting 2nd dimension
  size_t s1 = a.size1_, s2 = a.size2_;
#elif MEM_ORDER == VERTICAL
  // splitting 1st dimension
  size_t s1 = a.size2_, s2 = a.size1_;
#endif
  // splitting 2nd dimension
  if (matrices.size() != s1) matrices.resize(s1);
  size_t numel = mapsize[0] * mapsize[1];
  mexAssert(s2 % numel == 0, "In 'Mat::InitMaps' the matrix sizes do not correspond");
  size_t outputmaps = s2 / numel;
  for (size_t k = 0; k < matrices.size(); ++k) {
    if (matrices[k].size() != outputmaps) matrices[k].resize(outputmaps);
    size_t ind = 0;
    for (size_t j = 0; j < outputmaps; ++j) {
      matrices[k][j].attach(a.data_ + k*s2 + ind, mapsize[0], mapsize[1]);
      ind += numel;
    }    
  }
}

// layer transformation functions

void Prod(const Mat &a, bool a_tr, const Mat &b, bool b_tr, Mat &c) {
  
  size_t as1, as2, bs1, bs2;
  Mat al, bl, cl;
  if (!a_tr) { // a
    as1 = a.size1_; as2 = a.size2_;    
    al = Trans(a);
  } else { // aT
    as1 = a.size2_; as2 = a.size1_;
    al = a;
  }
  if (!b_tr) { //b
    bs1 = b.size1_; bs2 = b.size2_;
    bl = b;
  } else { //bT
    bs1 = b.size2_; bs2 = b.size1_;
    bl = Trans(b);
  }
  mexAssert(as2 == bs1, "In Prod the sizes of matrices do not correspond");  
  cl.init(as1, bs2, 0);
  //#if USE_MULTITHREAD == 1
  //  #pragma omp parallel for
  //#endif
  for (int k = 0; k < as2; k++) {    
    for (int i = 0; i < as1; i++) {      
      for (int j = 0; j < bs2; j++) {
        //#if USE_MULTITHREAD == 1
        //  #pragma omp atomic
        //#endif
        cl(i, j) += al(k, i) * bl(k, j);
      }
    }
  }
  c = std::move(cl);
}

void Filter(const Mat &image, const Mat &filter, 
            const std::vector<size_t> padding, Mat &filtered) {
  mexAssert(filtered.size1_ == image.size1_ + 2*padding[0] + 1 - filter.size1_ &&   
            filtered.size2_ == image.size2_ + 2*padding[1] + 1 - filter.size2_,
            "In 'Filter' the parameters do not correspond each other");
  Mat cache_mat(filtered.size1_, filtered.size2_);
  if (padding[0] == 0 && padding[1] == 0) {
    for (size_t i = 0; i < filtered.size1_; ++i) {
      for (size_t j = 0; j < filtered.size2_; ++j) {
        cache_mat(i, j) = 0;
        for (size_t u = 0; u < filter.size1_; ++u) {
          for (size_t v = 0; v < filter.size2_; ++v) {
            cache_mat(i, j) += filter(u, v) * image(i + u, j + v);
          }        
        }
      }
    }
  } else {
    int padding1 = (int) padding[0];
    int padding2 = (int) padding[1];
    int filtersize1 = (int) filter.size1_;
    int filtersize2 = (int) filter.size2_;
    int offset1 = (int) filtered.size1_ - 1 + filtersize1 - padding1;
    int offset2 = (int) filtered.size2_ - 1 + filtersize2 - padding2;        
    for (int i = 0; i < filtered.size1_; ++i) {
      for (int j = 0; j < filtered.size2_; ++j) {
        cache_mat(i, j) = 0;
        int minu = std::max(padding1 - i, 0);
        int minv = std::max(padding2 - j, 0);
        int maxu = std::min(offset1 - i, filtersize1);
        int maxv = std::min(offset2 - j, filtersize2);
        for (int u = minu; u < maxu; ++u) {
          for (int v = minv; v < maxv; ++v) {
            cache_mat(i, j) += filter(u, v) * image(i + u - padding1, j + v - padding2);
          }        
        }
      }
    }
  }
  filtered = std::move(cache_mat); 
}

void Transform(const Mat &image, const std::vector<ftype> &shift,
               const std::vector<ftype> &scale, const std::vector<bool> &mirror, 
               ftype angle, ftype defval, Mat &transformed) {
  ftype m1 = (ftype) image.size1_ / 2 - 0.5;
  ftype m2 = (ftype) image.size2_ / 2 - 0.5;  
  ftype n1 = (ftype) transformed.size1_ / 2 - 0.5;
  ftype n2 = (ftype) transformed.size2_ / 2 - 0.5;
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
      int xf1 = std::floor(x1);
      int xf2 = std::floor(x2);
      if (0 <= xf1 && xf1 + 1 <= image.size1_ - 1 &&
          0 <= xf2 && xf2 + 1 <= image.size2_ - 1) {        
        ftype vl = (x1 - xf1) * image(xf1+1, xf2) + (xf1 + 1 - x1) * image(xf1, xf2);
        ftype vh = (x1 - xf1) * image(xf1+1, xf2+1) + (xf1 + 1 - x1) * image(xf1, xf2+1);
        transformed(i, j) = (x2 - xf2) * vh + (xf2 + 1 - x2) * vl;
      } else {
        transformed(i, j) = defval;
      } 
    }
  }  
}

void MeanScale(const Mat &image, const std::vector<size_t> &scale,
               const std::vector<size_t> &stride, Mat &scaled) {
  mexAssert(scaled.size1_ == ceil((ftype) image.size1_ / stride[0]) &&
            scaled.size2_ == ceil((ftype) image.size2_ / stride[1]),
            "In 'MeanScale' the parameters do not correspond each other");
  scaled.assign(0);
  for (size_t i = 0; i < scaled.size1_; ++i) {
    for (size_t j = 0; j < scaled.size2_; ++j) {
      size_t maxu = std::min(scale[0], image.size1_ - i*stride[0]);
      size_t maxv = std::min(scale[1], image.size2_ - j*stride[1]);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          scaled(i, j) += image(i*stride[0]+u, j*stride[1]+v);
        }
      }
      scaled(i, j) /= (maxu * maxv);
    }
  }  
}

void MeanScaleDer(const Mat &image, const std::vector<size_t> &scale,
                  const std::vector<size_t> &stride, Mat &scaled) {
  mexAssert(image.size1_ == ceil((ftype) scaled.size1_ / stride[0]) && 
            image.size2_ == ceil((ftype) scaled.size2_ / stride[1]),
            "In 'MeanScaleDer' the parameters do not correspond each other");  
  scaled.assign(0);
  for (size_t i = 0; i < image.size1_; ++i) {
    for (size_t j = 0; j < image.size2_; ++j) {
      size_t maxu = std::min(scale[0], scaled.size1_-i*stride[0]);
      size_t maxv = std::min(scale[1], scaled.size2_-j*stride[1]);      
      ftype scaled_val = image(i, j) / (maxu * maxv);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          scaled(i*stride[0]+u, j*stride[1]+v) += scaled_val;
        }
      }      
    }
  }
}

void MaxScale(const Mat &image, const std::vector<size_t> &scale,
              const std::vector<size_t> &stride, Mat &scaled) {
  mexAssert(scaled.size1_ == ceil((ftype) image.size1_ / stride[0]) &&
            scaled.size2_ == ceil((ftype) image.size2_ / stride[1]),
            "In 'MaxScale' the parameters do not correspond each other");
  for (size_t i = 0; i < scaled.size1_; ++i) {
    for (size_t j = 0; j < scaled.size2_; ++j) {
      size_t maxu = std::min(scale[0], image.size1_-i*stride[0]);
      size_t maxv = std::min(scale[1], image.size2_-j*stride[1]);
      scaled(i, j) = image(i*stride[0], j*stride[1]);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          if (scaled(i, j) < image(i*stride[0]+u, j*stride[1]+v)) {
            scaled(i, j) = image(i*stride[0]+u, j*stride[1]+v);
          }
        }
      }      
    }
  }  
}

void MaxScaleDer(const Mat&image, const Mat &val, const Mat &prev_val,
                 const std::vector<size_t> &scale, const std::vector<size_t> &stride,
                 Mat &scaled) {
  mexAssert(image.size1_ == ceil((ftype) scaled.size1_ / stride[0]) && 
            image.size2_ == ceil((ftype) scaled.size2_ / stride[1]),
            "In 'MaxScaleDer' the parameters do not correspond each other");  
  scaled.assign(0);
  for (size_t i = 0; i < image.size1_; ++i) {
    for (size_t j = 0; j < image.size2_; ++j) {
      size_t maxu = std::min(scale[0], scaled.size1_-i*stride[0]);
      size_t maxv = std::min(scale[1], scaled.size2_-j*stride[1]);      
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          if (prev_val(i*stride[0]+u, j*stride[1]+v) == val(i, j)) {
            scaled(i*stride[0]+u, j*stride[1]+v) += image(i, j);
          }
        }
      }      
    }
  }
}
                 
void MaxTrim(const Mat &image, std::vector<size_t> &coords, Mat &trimmed) {
  
  mexAssert(trimmed.size1_ <= image.size1_ && trimmed.size2_ <= image.size2_,
            "In 'MaxTrim' the trimmed image is larger than original");
  size_t lv = std::floor((ftype) (trimmed.size1_ - 1)/2);  
  size_t lh = std::floor((ftype) (trimmed.size2_ - 1)/2);  
  
  ftype maxval = image(lv, lh);
  coords[0] = lv;
  coords[1] = lh;
  for (size_t i = lv; i < lv + image.size1_ - trimmed.size1_; ++i) {
    for (size_t j = lh; j < lh + image.size2_ - trimmed.size2_; ++j) {      
      if (maxval < image(i, j)) {
        maxval = image(i, j);
        coords[0] = i;
        coords[1] = j;
      }
    }
  }  
  for (size_t i = 0; i < trimmed.size1_; ++i) {
    for (size_t j = 0; j < trimmed.size2_; ++j) {
      trimmed(i, j) = image(coords[0]+i-lv, coords[1]+j-lh);      
    }
  }  
}

void MaxTrimDer(const Mat &image, const std::vector<size_t> &coords, Mat &restored) {
  mexAssert(restored.size1_ >= image.size1_ && restored.size2_ >= image.size2_,
            "In 'Mat::MaxRestore' the restored image is smaller than original");
  size_t lv = std::floor((ftype) (image.size1_ - 1)/2);  
  size_t lh = std::floor((ftype) (image.size2_ - 1)/2);  
  for (size_t i = 0; i < restored.size1_; ++i) {
    for (size_t j = 0; j < restored.size2_; ++j) {
      if (coords[0] <= i + lv && i + lv < coords[0] + image.size1_ &&
          coords[1] <= j + lh && j + lh < coords[1] + image.size2_) {
        restored(i, j) = image(i+lv-coords[0], j+lh-coords[1]);
      } else {
        restored(i, j) = 0;
      }
    }
  }
}
