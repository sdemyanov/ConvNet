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

#ifndef _MAT_CPU_H_
#define _MAT_CPU_H_

#include "settings.h"
#include "mex_print.h"
#include <array>
#include <vector>
#include <algorithm>
#include <random>
//#include <climits>

typedef std::array<int, 2> Pair;
typedef std::array<int, 4> Dim;

class MatCPU {

protected:
//public:
  /* pointer to the first matrix element */
  ftype *data_;

  /* define the maximum values for first and second indices accordingly */
  size_t size1_, size2_;

  /* indicates the major dimension in this matrix */
  bool order_;

  /* owner_ = true means that memory was allocated for this matrix */
  bool owner_;

private:
  inline size_t index(size_t i, size_t j) const {
    if (order_ == false) {
      return j * size1_ + i;
    } else {
      return i * size2_ + j;
    }
  }
  inline ftype& data(size_t ind) { return data_[ind]; }
  inline const ftype& data(size_t ind) const { return data_[ind]; }

public:

  // data access
  inline ftype& operator () (size_t i, size_t j) { return data_[index(i, j)]; }
  inline const ftype& operator () (size_t i, size_t j) const { return data_[index(i, j)]; }
  inline ftype& data(size_t i, size_t j) { return data_[index(i, j)]; }
  inline const ftype& data(size_t i, size_t j) const { return data_[index(i, j)]; }
  inline bool empty() const { return (data_ == NULL); }

  // public
  inline size_t size1() const { return size1_; }
  inline size_t size2() const { return size2_; }
  inline size_t size() const { return size1_ * size2_; }
  inline bool order() const { return order_; }

  //inline ftype operator () (size_t ind) const { return data_[ind]; }
  //MatCPU operator () (size_t ind);


  // memory functions
  MatCPU();
  MatCPU(size_t size1, size_t size2);
  MatCPU(const MatCPU &b);
  MatCPU(MatCPU &&b);
  ~MatCPU();
  MatCPU& init();
  MatCPU& operator = (const MatCPU &b);
  MatCPU& resize(size_t size1, size_t size2);
  MatCPU& reshape(size_t size1, size_t size2);
  MatCPU& set_order(bool order);
  MatCPU& attach(const MatCPU &b);
  MatCPU& attach(const MatCPU &b, size_t offset, size_t size1, size_t size2, bool order);
  MatCPU& attach(ftype *ptr, size_t size1, size_t size2);
  MatCPU& attach(ftype *ptr, size_t size1, size_t size2, bool order);
  MatCPU& clear();
  friend void Swap(MatCPU &a, MatCPU &b);

  // data functions
  MatCPU& ident();
  MatCPU& assign(ftype c);
  MatCPU& assign(const std::vector<ftype> &vect);
  MatCPU& operator += (const MatCPU &b);
  MatCPU& operator -= (const MatCPU &b);
  MatCPU& operator *= (const MatCPU &b);
  MatCPU& operator /= (const MatCPU &b);
  MatCPU& operator += (ftype c);
  MatCPU& operator -= (ftype c);
  MatCPU& operator *= (ftype c);
  MatCPU& operator /= (ftype c);
  MatCPU& Reorder(bool order);

  // friend functions
  friend void Sum(const MatCPU &a, MatCPU &vect, int dim);
  friend void Mean(const MatCPU &a, MatCPU &vect, int dim);
  friend void Trans(const MatCPU &a, MatCPU &b);
  friend void Shuffle(MatCPU &a, MatCPU &b);

  // const functions
  ftype sum() const;
  bool hasZeros() const;
};

#endif
