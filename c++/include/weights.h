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

#ifndef _WEIGHTS_H_
#define _WEIGHTS_H_

#include "mat.h"
#include "params.h"

class Weights {
  
public:
  Weights() {};
  ~Weights() { Clear(); };
  void Init(const Mat &weights);
  void Attach(Weights &weights, size_t offset, size_t size1, size_t size2, bool order);  
  void Update(const Params &params, size_t epoch, bool isafter);
  void Clear();
  inline Mat& get() { return weights_; }
  inline const Mat& get() const { return weights_; }
  inline Mat& der() { return weights_der_; }
  inline Mat& der2() { return weights_der2_; }
  inline std::vector<size_t> size() const { return size_; }  
  
private:
  Mat weights_;
  Mat weights_der_;
  Mat weights_der2_;
  Mat weights_der_prev_;
  Mat weights_learn_coefs_; 
  std::vector<size_t> size_;
  
};

#endif