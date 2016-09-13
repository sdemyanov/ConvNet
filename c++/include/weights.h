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

#ifndef _WEIGHTS_H_
#define _WEIGHTS_H_

#include "mat_gpu.h"
#include "params.h"

class Weights {

private:
  MatGPU weights_;
  MatGPU weights_der_;
  MatGPU weights_der2_;
  MatGPU weights_der_prev_;
  Dim dims_;

  void attach(Weights &w, size_t offset);

public:
  Weights();
  ~Weights() { Clear(); };
  void Init(const MatCPU &w);
  void AttachFilters(Weights &w, size_t offset);
  void AttachBiases(Weights &w, size_t offset);
  void RestoreOrder();
  size_t Num() const;
  void Update(const Params &params);
  void Clear();

  inline MatGPU& get() { return weights_; }
  inline const MatGPU& get() const { return weights_; }
  inline MatGPU& der() { return weights_der_; }
  inline MatGPU& der2() { return weights_der2_; }
  inline Dim& dims() { return dims_; }
  inline const Dim& dims() const { return dims_; }
  inline int& dims(int i) { return dims_[i]; }
  inline const int& dims(int i) const { return dims_[i]; }

};

#endif
