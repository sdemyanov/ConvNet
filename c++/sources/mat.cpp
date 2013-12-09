/*
Copyright (C) 2013 Sergey Demyanov. 
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

void Mat::Filter(const Mat& filter, Mat &filtered, bool type) const {  
  if (!type) { // valid filtration
    filtered.resize(mat_.size1()+1-filter.size1(), mat_.size2()+1-filter.size2());
    for (size_t i = 0; i < filtered.size1(); ++i) {
      for (size_t j = 0; j < filtered.size2(); ++j) {
        filtered(i, j) = 0;
        for (size_t u = 0; u < filter.size1(); ++u) {
          for (size_t v = 0; v < filter.size2(); ++v) {
            filtered(i, j) += filter(u, v) * mat_(i+u, j+v);
          }        
        }
      }
    }
  } else { // full filtration    
    filtered.resize(mat_.size1()+filter.size1()-1, mat_.size2()+filter.size2()-1);
    for (long i = 0; i < filtered.size1(); ++i) {
      for (long j = 0; j < filtered.size2(); ++j) {
        filtered(i, j) = 0;
        size_t minu = std::max((long) 0, (long) filter.size1() - i - 1);
        size_t minv = std::max((long) 0, (long) filter.size2() - j - 1);
        size_t maxu = std::min((long) filter.size1(), (long) filtered.size1() - i);
        size_t maxv = std::min((long) filter.size2(), (long) filtered.size2() - j);
        for (size_t u = minu; u < maxu; ++u) {
          for (size_t v = minv; v < maxv; ++v) {
            filtered(i, j) += filter(u, v) * 
              mat_(i+u+1-filter.size1(), j+v+1-filter.size2());            
          }        
        }
      }
    }
  }
}

Mat& Mat::Sigmoid() {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      mat_(i, j) = 1 / (1 + exp(-mat_(i, j)));
    }
  }
  return *this;
}

Mat& Mat::SigmDer(const Mat& a) {
  mexAssert(mat_.size1() == a.size1() && mat_.size2() == a.size2(), 
    "In 'Mat::SigmDer' the matrices are of the different size");
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      mat_(i, j) *= a(i, j) * (1 - a(i, j));
    }
  }
  return *this;
}  

void Mat::MeanScale(const std::vector<size_t> &scale, Mat &scaled) const {

  for (size_t i = 0; i < scaled.size1(); ++i) {
    for (size_t j = 0; j < scaled.size2(); ++j) {
      size_t maxu = std::min(scale[0], mat_.size1() - i*scale[0]);
      size_t maxv = std::min(scale[1], mat_.size2() - j*scale[1]);
      scaled(i, j) = 0;
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          scaled(i, j) += mat_(i*scale[0]+u, j*scale[1]+v);
        }
      }
      scaled(i, j) /= (maxu * maxv);
    }
  }
}

void Mat::MeanScaleDer(const std::vector<size_t> &scale, Mat &scaled) const {

  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      size_t maxu = std::min(scale[0], scaled.size1()-i*scale[0]);
      size_t maxv = std::min(scale[1], scaled.size2()-j*scale[1]);      
      double scaled_val = mat_(i, j) / (maxu * maxv);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          scaled(i*scale[0]+u, j*scale[1]+v) = scaled_val;
        }
      }      
    }
  }
}

void Mat::MaxScale(const std::vector<size_t> &scale, Mat &scaled) const {
  
  for (size_t i = 0; i < scaled.size1(); ++i) {
    for (size_t j = 0; j < scaled.size2(); ++j) {
      size_t maxu = std::min(scale[0],  mat_.size1()-i*scale[0]);
      size_t maxv = std::min(scale[1],  mat_.size2()-j*scale[1]);
      scaled(i, j) = mat_(i*scale[0], j*scale[1]);
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          if (scaled(i, j) < mat_(i*scale[0]+u, j*scale[1]+v)) {
            scaled(i, j) = mat_(i*scale[0]+u, j*scale[1]+v);
          }
        }
      }      
    }
  }
}

void Mat::MaxScaleDer(const std::vector<size_t> &scale, const Mat &val, const Mat &prev_val, Mat &scaled) const {
  
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      size_t maxu = std::min(scale[0], scaled.size1()-i*scale[0]);
      size_t maxv = std::min(scale[1], scaled.size2()-j*scale[1]);      
      for (size_t u = 0; u < maxu; ++u) {
        for (size_t v = 0; v < maxv; ++v) {          
          if (prev_val(i*scale[0]+u, j*scale[1]+v) == val(i, j)) {
            scaled(i*scale[0]+u, j*scale[1]+v) = mat_(i, j);
          } else {
            scaled(i*scale[0]+u, j*scale[1]+v) = 0;
          }
        }
      }      
    }
  }
}

void Mat::SubMat(const std::vector<size_t> ind, size_t dim, Mat &submat) const {
  
  mexAssert(ind.size() > 0, "In SubMat the index vector is empty");
  size_t minind = *(std::min_element(ind.begin(), ind.end()));
  mexAssert(minind >= 0, "In SubMat one of the indices is less than zero");    
  if (dim == 1) {
    size_t maxind = *(std::max_element(ind.begin(), ind.end()));    
    mexAssert(maxind < mat_.size1(), "In SubMat one of the indices is larger than the array size");    
    submat.resize(ind.size(), mat_.size2());
    for (size_t i = 0; i < ind.size(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        submat(i, j) = mat_(ind[i], j);
      }
    }
  } else if (dim == 2) {
    size_t maxind = *(std::max_element(ind.begin(), ind.end()));    
    mexAssert(maxind < mat_.size2(), "In SubMat one of the indices is larger than the array size");
    submat.resize(mat_.size1(), ind.size());
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < ind.size(); ++j) {
        submat(i, j) = mat_(i, ind[j]);
      }
    }    
  } else {
    mexAssert(false, "In Mat::SubMat the second parameter must be either 1 or 2");
  }
}

double Mat::Sum() const {
  double matsum = 0;
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      matsum += mat_(i, j);
    }
  }      
  return matsum;  
}

void Mat::Sum(size_t dim, Mat &vect) const {
  
  if (dim == 1) {
    vect.init(1, mat_.size2(), 0);
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        vect(0, j) += mat_(i, j);
      }
    }    
  } else if (dim == 2) {    
    vect.init(mat_.size1(), 1, 0);
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        vect(i, 0) += mat_(i, j);
      }     
    }    
  } else {
    mexAssert(false, "In Mat::Sum the dimension parameter must be either 1 or 2");
  }  
}

void Mat::Mean(size_t dim, Mat &vect) const {
  Sum(dim, vect);  
  if (dim == 1) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      vect(0, j) /= mat_.size1();
    }
  } else if (dim == 2) {    
    for (size_t i = 0; i < mat_.size1(); ++i) {
      vect(i, 0) /= mat_.size2();
    }    
  } else {
    mexAssert(false, "In Mat::Mean the dimension parameter must be either 1 or 2");
  }  
}

std::vector<size_t> Mat::MaxInd(size_t dim) const {
  std::vector<size_t> arrmax;
  if (dim == 1) {
    arrmax.assign(mat_.size2(), 0);    
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        if (mat_(i, j) > mat_(arrmax[j], j)) {
          arrmax[j] = i;
        }        
      }
    }    
  } else if (dim == 2) {
    arrmax.assign(mat_.size1(), 0);
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        if (mat_(i, j) > mat_(i, arrmax[i])) {
          arrmax[i] = j;
        }
      }     
    }    
  } else {
    mexAssert(false, "In Mat::MaxInd the dimension parameter must be either 1 or 2");
  }
  return arrmax;
}

Mat::Mat(const std::vector<size_t> &newsize) {
  mat_.resize(newsize[0], newsize[1]);
}

Mat::Mat(size_t size1, size_t size2) {
  mat_.resize(size1, size2);
}

Mat& Mat::resize(const std::vector<size_t> &newsize) {
  mat_.resize(newsize[0], newsize[1]);
  return *this;
}

Mat& Mat::resize(size_t size1, size_t size2) {
  mat_.resize(size1, size2);
  return *this;
}

Mat& Mat::init(const std::vector<size_t> &newsize, double val) {
  mat_.resize(newsize[0], newsize[1]);
  assign(val);
  return *this;
}

Mat& Mat::init(size_t size1, size_t size2, double val) {
  mat_.resize(size1, size2);
  assign(val);
  return *this;
}

Mat& Mat::assign(double val) {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      mat_(i, j) = val;
    }
  }
  return *this;
}

Mat& Mat::Rand() {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      mat_(i, j) = (double) rand() / RAND_MAX;
    }
  }
  return *this;
}

Mat& Mat::AddVect(const Mat &vect, size_t dim) {
  
  if (dim == 1) {
    mexAssert(vect.size2() == 1, "In 'Mat::AddVect' the second dimension must be 1"); 
    mexAssert(mat_.size1() == vect.size1(),
      "In 'Mat::AddVect' the second dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        mat_(i, j) += vect(i, 0);
      }
    }   
  } else if (dim == 2) {
    mexAssert(vect.size1() == 1, "In 'Mat::AddVect' the first dimension must be 1"); 
    mexAssert(mat_.size2() == vect.size2(),
      "In 'Mat::AddVect' the first dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        mat_(i, j) += vect(0, j);
      }
    }
  } else {
    mexAssert(false, "In Mat::AddVect the dimension parameter must be either 1 or 2");
  }
  return *this;
}

Mat& Mat::MultVect(const std::vector<double> &vect, size_t dim) {
  
  if (dim == 1) {
    mexAssert(mat_.size1() == vect.size(),
      "In 'Mat::MultVect' the second dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        mat_(i, j) *= vect[i];
      }
    }   
  } else if (dim == 2) {
    mexAssert(mat_.size2() == vect.size(),
      "In 'Mat::MultVect' the first dimension of matrix and length of vector are of the different size");
    for (size_t i = 0; i < mat_.size1(); ++i) {
      for (size_t j = 0; j < mat_.size2(); ++j) {
        mat_(i, j) *= vect[j];
      }
    }
  } else {
    mexAssert(false, "In Mat::MultVect the dimension parameter must be either 1 or 2");
  }
  return *this;
}

Mat& Mat::ReshapeFrom(const std::vector< std::vector<Mat> > &squeezed) {
  // stretches 1, 3 and 4 dimensions size_to the 1st one. The 2nd dimension stays the same.
  size_t outputmaps = squeezed.size();
  mexAssert(outputmaps > 0, "In 'Mat::ReshapeFrom' the number of output maps is zero");
  size_t batchsize = squeezed[0].size();
  mexAssert(batchsize > 0, "In 'Mat::ReshapeFrom' the number of batches is zero");
  size_t mapsize[2];
  mapsize[0] = squeezed[0][0].size1();
  mapsize[1] = squeezed[0][0].size2();
  size_t numel = mapsize[0] * mapsize[1];
  mat_.resize(outputmaps * numel, batchsize);  
  for (size_t j = 0; j < outputmaps; ++j) {
    for (size_t k = 0; k < batchsize; ++k) {
      mexAssert(squeezed[j][k].size1() == mapsize[0], "In 'Mat::ReshapeFrom' the first dimension is not constant");
      mexAssert(squeezed[j][k].size2() == mapsize[1], "In 'Mat::ReshapeFrom' the second dimension is not constant");
      for (size_t u = 0; u < mapsize[0]; ++u) {
        for (size_t v = 0; v < mapsize[1]; ++v) {            
          size_t ind = j*numel + u*mapsize[1] + v;
          mat_(ind, k) = squeezed[j][k](u, v);
        }
      }
    }      
  }
  return *this;
}

void Mat::ReshapeTo(std::vector< std::vector<Mat> > &squeezed,
                    size_t outputmaps, size_t batchsize, 
                    const std::vector<size_t> &mapsize) const {
  size_t numel = mapsize[0] * mapsize[1];
  mexAssert(outputmaps * numel == mat_.size1(), "In 'Mat::ReshapeTo' the first dimension do not correspond the parameters");
  mexAssert(batchsize == mat_.size2(), "In 'Mat::ReshapeTo' the second dimension do not correspond the parameters");
  squeezed.resize(outputmaps);
  for (size_t j = 0; j < outputmaps; ++j) {
    squeezed[j].resize(batchsize);
    for (size_t k = 0; k < batchsize; ++k) {
      squeezed[j][k].resize(mapsize);
      for (size_t u = 0; u < mapsize[0]; ++u) {
        for (size_t v = 0; v < mapsize[1]; ++v) {            
          size_t ind = j*numel + u*mapsize[1] + v;
          squeezed[j][k](u, v) = mat_(ind, k);
        }
      }
    }      
  }
}

void Mat::MaxTrim(Mat &trimmed, std::vector<size_t> &coords) const {
  
  mexAssert(trimmed.size1() <= mat_.size1(), "In 'Mat::Trim' the trimmed image is larger than original");
  mexAssert(trimmed.size2() <= mat_.size2(), "In 'Mat::Trim' the trimmed image is larger than original");
  size_t lv = std::floor((double) (trimmed.size1() - 1)/2);  
  size_t lh = std::floor((double) (trimmed.size2() - 1)/2);  
  
  double maxval = mat_(lv, lh);
  coords[0] = lv;
  coords[1] = lh;
  for (size_t i = lv; i < lv + mat_.size1() - trimmed.size1(); ++i) {
    for (size_t j = lh; j < lh + mat_.size2() - trimmed.size2(); ++j) {      
      if (maxval < mat_(i, j)) {
        maxval = mat_(i, j);
        coords[0] = i;
        coords[1] = j;
      }
    }
  }  
  for (size_t i = 0; i < trimmed.size1(); ++i) {
    for (size_t j = 0; j < trimmed.size2(); ++j) {
      trimmed(i, j) = mat_(coords[0]+i-lv, coords[1]+j-lh);      
    }
  }  
}

void Mat::MaxRestore(Mat &restored, const std::vector<size_t> &coords) const {
  
  mexAssert(restored.size1() >= mat_.size1(), "In 'Mat::MaxRestore' the restored image is smaller than original");
  mexAssert(restored.size2() >= mat_.size2(), "In 'Mat::MaxRestore' the restored image is smaller than original");
  size_t lv = std::floor((double) (mat_.size1() - 1)/2);  
  size_t lh = std::floor((double) (mat_.size2() - 1)/2);  
  
  for (size_t i = 0; i < restored.size1(); ++i) {
    for (size_t j = 0; j < restored.size2(); ++j) {
      if (coords[0] <= i + lv && i + lv < coords[0] + mat_.size1() &&
          coords[1] <= j + lh && j + lh < coords[1] + mat_.size2()) {
        restored(i, j) = mat_(i+lv-coords[0], j+lh-coords[1]);
      } else {
        restored(i, j) = 0;
      }
    }
  }
}

Mat& Mat::FromVect(const std::vector<double> &vect, const std::vector<size_t> &newsize) {
  mexAssert(vect.size() == newsize[0] * newsize[1], 
    "In 'Mat::FromVect' the vector and sizes do not correspond");
  mat_.resize(newsize[0], newsize[1]);
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      mat_(i, j) = vect[i*mat_.size2()+j];
    }
  }
  return *this;
}
  
std::vector<double> Mat::ToVect() const {
  std::vector<double> vect(mat_.size1() * mat_.size2());  
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      vect[i*mat_.size2()+j] = mat_(i, j);
    }
  }
  return vect;
}

Mat& Mat::ElemProd(const Mat &a) {
  mexAssert(mat_.size1() == a.size1() && mat_.size2() == a.size2(), 
    "In 'Mat::ElemProd' the matrices are of the different size");
  mat_ = element_prod(mat_, a.mat_);  
  return *this;
}

Mat& Mat::Trans() {
  mat_ = trans(mat_);
  return *this;
}

Mat& Mat::Sign() {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {
      if (mat_(i, j) > 0) {
        mat_(i, j) = 1;
      } else if (mat_(i, j) < 0) {
        mat_(i, j) = -1;
      } else {
        mat_(i, j) = 0;
      }
    }
  }
  return *this;
}

Mat& Mat::ElemMax(double a) {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      if (mat_(i, j) < a) mat_(i, j) = a;
    }
  }
  return *this;
}

Mat& Mat::CondAssign(const Mat &condmat, double threshold, bool incase, double a) {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      if (incase == (condmat(i, j) > threshold)) mat_(i, j) = a; // xor
    }
  }
  return *this;
}

Mat& Mat::CondAdd(const Mat &condmat, double threshold, bool incase, double a) {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      if (incase == (condmat(i, j) > threshold)) mat_(i, j) += a; // xor
    }
  }
  return *this;
}

Mat& Mat::CondProd(const Mat &condmat, double threshold, bool incase, double a) {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      if (incase == (condmat(i, j) > threshold)) mat_(i, j) *= a; // xor
    }
  }
  return *this;
}

double& Mat::operator () (size_t ind) {
  if (mat_.size1() == 1) {    
    return mat_(0, ind);
  } else if (mat_.size2() == 1) {    
    return mat_(ind, 0);
  } else {
    mexAssert(false, "In 'Mat::(ind)' matrix is not really a vector");     
  }  
}

const double& Mat::operator () (size_t ind) const {
  if (mat_.size1() == 1) {    
    return mat_(0, ind);
  } else if (mat_.size2() == 1) {    
    return mat_(ind, 0);
  } else {
    mexAssert(false, "In 'Mat::(ind)' matrix is not really a vector");     
  }
}

double& Mat::operator() (size_t ind1, size_t ind2) {
  return mat_(ind1, ind2);
}

const double& Mat::operator() (size_t ind1, size_t ind2) const {
  return mat_(ind1, ind2);
}

Mat& Mat::operator = (const Mat &mat) {
  mat_ = mat.mat_;
  return *this;
}

size_t Mat::size1() const {
  return mat_.size1();
}

size_t Mat::size2() const {
  return mat_.size2();
}

Mat& Mat::operator += (const Mat &a) {
  mexAssert(mat_.size1() == a.size1() && mat_.size2() == a.size2(),
  "In Mat::+= the sizes of matrices do not correspond");
  mat_ += a.mat_;
  return *this;
}

Mat& Mat::operator -= (const Mat &a) {
  mexAssert(mat_.size1() == a.size1() && mat_.size2() == a.size2(),
  "In Mat::+= the sizes of matrices do not correspond");
  mat_ -= a.mat_;
  return *this;
}

Mat& Mat::operator += (double a) {
  for (size_t i = 0; i < mat_.size1(); ++i) {
    for (size_t j = 0; j < mat_.size2(); ++j) {      
      mat_(i, j) += a;
    }
  }
  return *this;
}

Mat& Mat::operator -= (double a) {
  *this += -a;
  return *this;
}

Mat& Mat::operator *= (double a) {
  mat_ *= a;
  return *this;
}

Mat& Mat::operator /= (double a) {
  mat_ /= a;
  return *this;
}
  
void Trans(const Mat &a, Mat &c) {
  c.mat_.resize(a.size2(), a.size1());
  noalias(c.mat_) = trans(a.mat_);
}

void Prod(const Mat &a, const Mat &b, Mat &c) {
  mexAssert(a.size2() == b.size1(),
  "In Prod the sizes of matrices do not correspond");
  c.mat_.resize(a.size1(), b.size2());
  noalias(c.mat_) = prod(a.mat_, b.mat_);  
}

void Sum(const Mat &a, const Mat &b, Mat &c) {  
  mexAssert(a.size1() == b.size1() && a.size2() == b.size2(),
  "In Sum the sizes of matrices do not correspond");
  c.mat_.resize(a.size1(), b.size2());
  noalias(c.mat_) = a.mat_ + b.mat_;
}
