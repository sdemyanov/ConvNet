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

#include "mat_cpu.h"
#include <algorithm>
#include <cmath>

/*
MatCPU MatCPU::operator () (size_t ind) {
  MatCPU val_mat;
  val_mat.attach(&data(ind), 1, 1);
  return val_mat;
}*/

// memory functions

MatCPU::MatCPU() {
  init();
}

MatCPU::MatCPU(size_t size1, size_t size2) {
  init();
  resize(size1, size2);
}

MatCPU::MatCPU(const MatCPU &b) {
  init();
  if (b.empty()) return;
  resize(b.size1_, b.size2_);
  (*this) = b;
}

MatCPU::MatCPU(MatCPU &&b) {
  init();
  Swap(*this, b);
}

MatCPU::~MatCPU() {
  clear();
}

MatCPU& MatCPU::init() {
  data_ = NULL;
  size1_ = 0;
  size2_ = 0;
  order_ = kInternalOrder;
  owner_ = false;
  return *this;
}

// only this function supports order_ == true
MatCPU& MatCPU::operator = (const MatCPU &b) {
  //mexPrintMsg("Array const assignment");
  mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
    "In MatCPU::operator = the arrays have different size");
  if (order_ == b.order_) {
    for (size_t i = 0; i < size(); ++i) {
      data(i) = b.data(i);
    }
  } else if (b.order_ != kInternalOrder) { // order_ == kInternalOrder
    MatCPU br;
    br.attach(b.data_, b.size2_, b.size1_, kInternalOrder);
    Trans(br, *this);
  } else { // b.order_ == kInternalOrder, order_ == !kInternalOrder
    MatCPU br;
    br.attach(data_, size2_, size1_, kInternalOrder);
    Trans(b, br);
  }
  //mexPrintMsg("Array const assignment end");
  return *this;
}

MatCPU& MatCPU::resize(size_t size1, size_t size2) {
  //mexPrintMsg("Array resize");
  if (size1 * size2 != size()) {
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
  mexAssertMsg(size() == size1 * size2,
    "In MatCPU::reshape the sizes do not correspond");
  size1_ = size1;
  size2_ = size2;
  return *this;
}

MatCPU& MatCPU::set_order(bool order) {
  order_ = order;
  return *this;
}

MatCPU& MatCPU::attach(const MatCPU &b) {
  return attach(b.data_, b.size1_, b.size2_, b.order_);
}

MatCPU& MatCPU::attach(const MatCPU &b, size_t offset, size_t size1, size_t size2, bool order) {
  mexAssertMsg(b.size1_ == 1 || b.size2_ == 1, "In MatCPU::attach with offset one of sizes should be 1");
  mexAssertMsg(offset + size1 * size2 <= b.size(),
            "In MatCPU::attach the sizes don't correspond each other");
  return attach(b.data_ + offset, size1, size2, order);
}

MatCPU& MatCPU::attach(ftype *ptr, size_t size1, size_t size2) {
  return attach(ptr, size1, size2, kInternalOrder);
}

MatCPU& MatCPU::attach(ftype *ptr, size_t size1, size_t size2, bool order) {
  //mexAssertMsg(order == false, "In MatCPU::attach order should be always false");
  clear();
  data_ = ptr;
  size1_ = size1;
  size2_ = size2;
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

  bool order_tmp = b.order_;
  b.order_ = a.order_;
  a.order_ = order_tmp;

  bool owner_tmp = b.owner_;
  b.owner_ = a.owner_;
  a.owner_ = owner_tmp;
}

// data functions

MatCPU& MatCPU::ident() {
  mexAssertMsg(size1_ == size2_,
    "In 'MatCPU::ident' the matrix must be squared");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      if (i == j) {
        data(i, j) = 1;
      } else {
        data(i, j) = 0;
      }
    }
  }
  return *this;
}

MatCPU& MatCPU::assign(ftype c) {
  for (size_t i = 0; i < size(); ++i) {
    data(i) = c;
  }
  return *this;
}

MatCPU& MatCPU::assign(const std::vector<ftype> &vect) {
  mexAssertMsg(size1_ == 1 && size2_ == vect.size(),
    "In MatCPU::assign the sizes do not correspond");
  for (size_t i = 0; i < vect.size(); ++i) {
    data(i) = vect[i];
  }
  return *this;
}

MatCPU& MatCPU::operator += (const MatCPU &b) {
  mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
    "In MatCPU::+= the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) += b(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator -= (const MatCPU &b) {
  mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
    "In MatCPU::-= the sizes of matrices do not correspond");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) -= b(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator *= (const MatCPU &b) {
  mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
    "In 'MatCPU::*=' the matrices are of the different size");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) *= b(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator /= (const MatCPU &b) {
  mexAssertMsg(size1_ == b.size1_ && size2_ == b.size2_,
    "In 'MatCPU::/=' the matrices are of the different size");
  for (size_t i = 0; i < size1_; ++i) {
    for (size_t j = 0; j < size2_; ++j) {
      data(i, j) /= b(i, j);
    }
  }
  return *this;
}

MatCPU& MatCPU::operator += (ftype c) {
  for (size_t i = 0; i < size(); ++i) {
    data(i) += c;
  }
  return *this;
}

MatCPU& MatCPU::operator -= (ftype c) {
  for (size_t i = 0; i < size(); ++i) {
    data(i) -= c;
  }
  return *this;
}

MatCPU& MatCPU::operator *= (ftype c) {
  for (size_t i = 0; i < size(); ++i) {
    data(i) *= c;
  }
  return *this;
}

MatCPU& MatCPU::operator /= (ftype c) {
  for (size_t i = 0; i < size(); ++i) {
    data(i) /= c;
  }
  return *this;
}

MatCPU& MatCPU::Reorder(bool order) {
  //mexAssertMsg(order_ == order, "In MatCPU::reorder orders should be the same");
  if (order_ != order) {
    if (size1_ > 1 && size2_ > 1) {
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

// friend functions

void Sum(const MatCPU &a, MatCPU &vect, int dim) {

  if (dim == 1) {
    mexAssertMsg(vect.size1_ == 1 && vect.size2_ == a.size2_,
      "In Sum the sizes do not correspond each other");
    vect.assign(0);
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < a.size2_; ++j) {
        vect(0, j) += a(i, j);
      }
    }
  } else if (dim == 2) {
    mexAssertMsg(vect.size1_ == a.size1_ && vect.size2_ == 1,
      "In Sum the sizes do not correspond each other");
    vect.assign(0);
    for (size_t i = 0; i < a.size1_; ++i) {
      for (size_t j = 0; j < a.size2_; ++j) {
        vect(i, 0) += a(i, j);
      }
    }
  } else {
    mexAssertMsg(false, "In MatCPU Sum the dimension parameter must be either 1 or 2");
  }
}

void Mean(const MatCPU &a, MatCPU &vect, int dim) {
  Sum(a, vect, dim);
  if (dim == 1) {
    vect /= (ftype) a.size1_;
  } else if (dim == 2) {
    vect /= (ftype) a.size2_;
  } else {
    mexAssertMsg(false, "In MatCPU Mean the dimension parameter must be either 1 or 2");
  }
}

void Trans(const MatCPU &a, MatCPU &b) {
  // no resize to ensure that b.data_ is not relocated
  mexAssertMsg(a.size1_ == b.size2_ && a.size2_ == b.size1_,
            "In Trans the sizes of matrices do not correspond");
  mexAssertMsg(a.data_ != b.data_, "In Trans the matrices are the same");
  for (size_t i = 0; i < b.size1_; ++i) {
    for (size_t j = 0; j < b.size2_; ++j) {
      b(i, j) = a(j, i);
    }
  }
}

void Shuffle(MatCPU &a, MatCPU &b) {
  mexAssertMsg(a.order_ == true && b.order_ == true, "In Shuffle the orders should be true");
  mexAssertMsg(a.size1_ == b.size1_, "In Shuffle the sizes do not correspond");
  size_t train_num = a.size1_;
  //mexPrintMsg("train_num", train_num);
  std::vector<int> randind(train_num);
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


ftype MatCPU::sum() const {
  ftype matsum = 0;
  for (size_t i = 0; i < size(); ++i) {
    matsum += data(i);
  }
  return matsum;
}

bool MatCPU::hasZeros() const {
  for (size_t i = 0; i < size(); ++i) {
    if (data(i) == 0) return true;
  }
  return false;
}
