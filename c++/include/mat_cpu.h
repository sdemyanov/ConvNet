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

#ifndef _MAT_CPU_H_
#define _MAT_CPU_H_

#include "settings.h"
#include "mex_print.h"
#include <vector>
#include <algorithm>
#include <random>

/*
order = false -> size1 is major
order = true  -> size2 is major

Major means that address changes in this dimension first.
For a particular matrix the order is indicated by the order_ variable.
*/

// always false
static const bool kMatlabOrder = false; 

// indicates how maps and filters are stored in the memory
// within their containers, such as activ_mat_ and weights_.get()
static const bool kMapsOrder = false;

// indicates how containers are stored in the memory
#if USE_CUDNN == 1
  static const bool kDefaultOrder = true;
#else
  static const bool kDefaultOrder = false;
#endif
  
class MatCPU {

private:
  //static
  static std::default_random_engine _generator;  
                         
protected:
//public:
  /* pointer to the first matrix element */
  ftype *data_;
  
  /* define the maximum values for first and second indices accordingly */  
  size_t size1_, size2_;
  
  /* shift in memory between consecutive array elements,
     differs from 1 only for map and filter matrices */  
  size_t stride_; 
  
  /* indicates the major dimension in this matrix, should be false by default */
  bool order_;
  
  /* owner_ = true means that memory was allocated for this matrix */
  bool owner_;  

public:  

  // static
  static void InitRand(size_t seed);
  
  // data access
  // private
  inline size_t index(size_t ind) const { return ind * stride_; }
  // assumes that order_ == false for all matrices that use double index
  inline size_t index(size_t i, size_t j) const { return (j * size1_ + i) * stride_; }  
  inline ftype& operator () (size_t i, size_t j) { return data_[index(i, j)]; } 
  inline const ftype& operator () (size_t i, size_t j) const { return data_[index(i, j)]; }
  inline ftype& data(size_t i, size_t j) { return data_[index(i, j)]; }
  inline const ftype& data(size_t i, size_t j) const { return data_[index(i, j)]; }
  inline ftype& data(size_t ind) { return data_[index(ind)]; }
  inline const ftype& data(size_t ind) const { return data_[index(ind)]; }
  inline bool empty() const { return (data_ == NULL); }  
  
  // public
  inline size_t size1() const { return size1_; }
  inline size_t size2() const { return size2_; }  
  inline bool order() const { return order_; }
  //inline ftype operator () (size_t ind) const { return data_[index(ind)]; }  
  //MatCPU operator () (size_t ind);
  
  // memory functions
  MatCPU();
  MatCPU(const std::vector<size_t> &newsize);
  MatCPU(size_t size1, size_t size2);
  MatCPU(const MatCPU &a);
  MatCPU(MatCPU &&a);
  ~MatCPU();
  MatCPU& init();
  MatCPU& operator = (const MatCPU &a);
  MatCPU& resize(size_t size1, size_t size2);  
  MatCPU& reshape(size_t size1, size_t size2);    
  MatCPU& reorder(bool order, bool real);
  MatCPU& attach(const MatCPU &a);       
  MatCPU& attach(const MatCPU &a, size_t offset, size_t size1, size_t size2, bool order);
  MatCPU& attach(ftype *ptr, size_t size1, size_t size2);  
  MatCPU& attach(ftype *ptr, size_t size1, size_t size2, size_t stride, bool order);  
  MatCPU& clear();  
  friend void Swap(MatCPU &a, MatCPU &b);    
    
  // data functions
  MatCPU& assign(ftype val);
  MatCPU& rand();  
  MatCPU& randnorm();
  MatCPU& operator += (const MatCPU &a);
  MatCPU& operator -= (const MatCPU &a);
  MatCPU& operator *= (const MatCPU &a);
  MatCPU& operator /= (const MatCPU &a);
  MatCPU& operator += (ftype a);
  MatCPU& operator -= (ftype a);
  MatCPU& operator *= (ftype a);
  MatCPU& operator /= (ftype a);  
  MatCPU& Sign();
  MatCPU& Sqrt();
  MatCPU& Exp(); 
  MatCPU& Log();  
  MatCPU& SoftMax();
  MatCPU& SoftDer(const MatCPU &a);
  MatCPU& Sigmoid();
  MatCPU& SigmDer(const MatCPU &a);
  MatCPU& CondAssign(const MatCPU &condmat, bool incase, ftype threshold, ftype a);
  MatCPU& CondAdd(const MatCPU &condmat, bool incase, ftype threshold, ftype a);
  MatCPU& CondMult(const MatCPU &condmat, bool incase, ftype threshold, ftype a);
  MatCPU& AddVect(const MatCPU &vect, size_t dim);
  MatCPU& MultVect(const MatCPU &vect, size_t dim);
  MatCPU& Normalize(ftype norm);
  MatCPU& Validate();
  
  // friend functions
  friend void Sum(const MatCPU &a, MatCPU &vect, size_t dim);
  friend void Mean(const MatCPU &a, MatCPU &vect, size_t dim);  
  
  friend void Trans(const MatCPU &a, MatCPU &b);  
  friend void SubSet(MatCPU &a, MatCPU &b, size_t offset, bool dir);
  friend void Shuffle(MatCPU &a, MatCPU &b);
  friend void InitMaps(const MatCPU &a, const std::vector<size_t> &mapsize,
                       std::vector< std::vector<MatCPU> > &matrices);
  
  // layer transformation functions
  friend void Prod(const MatCPU &a, bool a_tr, const MatCPU &b, bool b_tr, MatCPU &c);
  friend void Filter(const MatCPU &image, const MatCPU &filter, 
                     const std::vector<size_t> padding, bool conv, MatCPU &filtered);
  friend void Transform(const MatCPU &image, const std::vector<ftype> &shift, 
                        const std::vector<ftype> &scale, const std::vector<bool> &mirror,
                        ftype angle, ftype defval, MatCPU &transformed);         
  friend void MeanScale(const MatCPU &image, const std::vector<size_t> &scale,
                        const std::vector<size_t> &stride, MatCPU &scaled);
  friend void MeanScaleDer(const MatCPU &scaled_der, const std::vector<size_t> &scale,
                           const std::vector<size_t> &stride, MatCPU &image_der);  
  friend void MaxScale(const MatCPU &image, const std::vector<size_t> &scale,
                       const std::vector<size_t> &stride, MatCPU &scaled);
  friend void MaxScaleDer(const MatCPU &image, const MatCPU &scaled, MatCPU &scaled_der, MatCPU &image_der, 
                          const std::vector<size_t> &scale, const std::vector<size_t> &stride, bool dir);
                         
  // const functions
  ftype sum() const;
  bool hasZeros() const;
  void write(ftype *vect) const; 
  
};

#endif