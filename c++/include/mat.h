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

#ifndef _MAT_H_
#define _MAT_H_

#include "ftype.h"
#include "mex_print.h"
#include <vector>
#include <algorithm>
#include <omp.h>

#define MEM_ORDER HORIZONTAL
//#define MEM_ORDER VERTICAL

#if MEM_ORDER == HORIZONTAL
  #define IND_IJ i * size2_ + j
#elif MEM_ORDER == VERTICAL
  #define IND_IJ j * size1_ + i
#endif

class Mat{

public:  
  
  inline ftype& operator () (size_t i, size_t j) { return data_[IND_IJ]; } 
  inline const ftype& operator () (size_t i, size_t j) const { return data_[IND_IJ]; }
  inline ftype& data(size_t i, size_t j) { return data_[IND_IJ]; }
  inline size_t size1() const { return size1_; }
  inline size_t size2() const { return size2_; }  
  inline bool isempty() const { return (data_ == NULL); }  
  
  Mat();
  Mat(const std::vector<size_t> &newsize);
  Mat(size_t size1, size_t size2);
  Mat(const Mat &a);
  Mat(Mat &&a);
  ~Mat();
  Mat& operator = (const Mat &a);
  Mat& operator = (Mat &&a);
  void clear();
  Mat& attach(ftype *vect, const std::vector<size_t> &newsize);
  Mat& attach(ftype *vect, size_t size1, size_t size2);
  Mat& attach(const Mat &a);  
  Mat& assign(ftype val);
  Mat& rand();
  void resize(const std::vector<size_t> &newsize);
  void resize(size_t size1, size_t size2);
  Mat& copy(const Mat &a);
  Mat& init(const std::vector<size_t> &newsize, ftype val);
  Mat& init(size_t size1, size_t size2, ftype val);
  
    
  void ToVect(ftype *vect) const;
  ftype Sum() const;
  ftype& operator () (size_t ind);
  const ftype& operator () (size_t ind) const;
  Mat& operator += (const Mat &a);
  Mat& operator -= (const Mat &a);
  Mat& operator *= (const Mat &a);
  Mat& operator /= (const Mat &a);
  Mat& operator += (ftype a);
  Mat& operator -= (ftype a);
  Mat& operator *= (ftype a);
  Mat& operator /= (ftype a);  
  Mat& Sign();
  Mat& SoftMax();
  Mat& SoftDer(const Mat& a);
  Mat& Sigmoid();
  Mat& SigmDer(const Mat& a);
  Mat& ElemMax(ftype a);
  Mat& CondAssign(const Mat &condmat, ftype threshold, bool incase, ftype a);
  Mat& CondAdd(const Mat &condmat, ftype threshold, bool incase, ftype a);
  Mat& CondProd(const Mat &condmat, ftype threshold, bool incase, ftype a);
  Mat& AddVect(const Mat &vect, size_t dim);
  Mat& MultVect(const Mat &vect, size_t dim);
  Mat& Normalize(ftype norm, ftype &oldnorm);
  Mat& CalcPercents();
  Mat& Validate();
                    
  // friend functions
  friend void Swap(Mat &a, Mat &b);
  friend Mat Sum(const Mat &a, size_t dim);
  friend Mat Mean(const Mat &a, size_t dim);  
  friend Mat Trans(const Mat &a);
  friend Mat SubMat(const Mat &a, const std::vector<size_t> &ind, size_t dim);  
  friend void InitMaps(const Mat &a, const std::vector<size_t> &mapsize,
                       std::vector< std::vector<Mat> > &reshaped);
  
  // layer transformation functions
  friend void Prod(const Mat &a, bool a_tr, const Mat &b, bool b_tr, Mat &c);
  friend void Filter(const Mat &image, const Mat &filter, 
                     const std::vector<size_t> padding, Mat &filtered);
  friend void Transform(const Mat &image, const std::vector<ftype> &shift, 
                        const std::vector<ftype> &scale, const std::vector<bool> &mirror,
                        ftype angle, ftype defval, Mat &transformed);         
  friend void MeanScale(const Mat &image, const std::vector<size_t> &scale,
                        const std::vector<size_t> &stride, Mat &scaled);
  friend void MeanScaleDer(const Mat &image, const std::vector<size_t> &scale,
                           const std::vector<size_t> &stride, Mat &scaled);  
  friend void MaxScale(const Mat &image, const std::vector<size_t> &scale,
                       const std::vector<size_t> &stride, Mat &scaled);
  friend void MaxScaleDer(const Mat &image, const Mat &val, const Mat &prev_val,
                          const std::vector<size_t> &scale, const std::vector<size_t> &stride,
                          Mat &scaled);
  friend void MaxTrim(const Mat &image, std::vector<size_t> &coords, Mat &trimmed);
  friend void MaxTrimDer(const Mat &image, const std::vector<size_t> &coords,
                         Mat &restored);  
  
private:
  ftype *data_;
  size_t size1_, size2_; 
  bool owner_;  
  
};

#endif