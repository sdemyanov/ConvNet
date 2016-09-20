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

#include "weights.h"

Weights::Weights() {
  dims_ = {0, 0, 0, 0};
}

void Weights::Init(const MatCPU &w) {
  mexAssert(w.size() < INT_MAX);
  dims_ = {(int) w.size1(), (int) w.size2(), 1, 1};
  if (Num() == 0) return;

  weights_.resize_filter(dims_);
  weights_der_.resize_filter(dims_);
  weights_der2_.resize_filter(dims_);
  weights_der_prev_.resize_filter(dims_);

  weights_ = w;
  weights_der_.assign(0);
  weights_der2_.assign(0);
  weights_der_prev_.assign(0);
}

void Weights::attach(Weights &w, size_t offset) {
  size_t size1 = dims_[0], size2 = dims_[1] * dims_[2] * dims_[3];
  weights_.attach(w.weights_, offset, size1, size2, kExternalOrder);
  weights_der_.attach(w.weights_der_, offset, size1, size2, kInternalOrder);
  weights_der2_.attach(w.weights_der2_, offset, size1, size2, kInternalOrder);
  weights_der_prev_.attach(w.weights_der_prev_, offset, size1, size2, kInternalOrder);
}

void Weights::AttachFilters(Weights &w, size_t offset) {

  if (Num() == 0) return;
  attach(w, offset);
  weights_.reshape_filter(dims_);
  weights_der_.reshape_filter(dims_);
  weights_der2_.reshape_filter(dims_);
  weights_der_prev_.reshape_filter(dims_);

  weights_.Reorder(kInternalOrder);
  weights_.ReorderMaps(kExternalOrder, kInternalOrder);
}

void Weights::AttachBiases(Weights &w, size_t offset) {

  if (Num() == 0) return;
  attach(w, offset);
  weights_.set_order(kInternalOrder);

  weights_.reshape_tensor(dims_);
  weights_der_.reshape_tensor(dims_);
  weights_der2_.reshape_tensor(dims_);
  weights_der_prev_.reshape_tensor(dims_);
}

void Weights::RestoreOrder() {
  weights_.ReorderMaps(kInternalOrder, kExternalOrder);
  weights_.Reorder(kExternalOrder);
}

size_t Weights::Num() const {
  return dims_[0] * dims_[1] * dims_[2] * dims_[3];
}

void Weights::Update(const Params &params) {
  if (params.beta_ > 0) {
    weights_der2_ *= params.beta_;
    weights_der_ += weights_der2_;
  }
  if (params.momentum_ > 0) {
    weights_der_prev_ *= params.momentum_;
    weights_der_ *= (1 - params.momentum_);
    weights_der_ += weights_der_prev_;
    weights_der_prev_ = weights_der_;
  }
  weights_der_ *= params.alpha_;
  if (params.decay_ > 0) {
    weights_ *= (1 - params.decay_ * params.alpha_);
  }
  // direction that decreases the error
  weights_ -= weights_der_;
  weights_der_.assign(0);
  weights_der2_.assign(0);
}

void Weights::Clear() {
  weights_.clear();
  weights_der_.clear();
  weights_der2_.clear();
  weights_der_prev_.clear();
}
