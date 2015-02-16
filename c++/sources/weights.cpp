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

#include "weights.h"

void Weights::Init(const Mat &weights) {
  size_.resize(2);
  size_[0] = weights.size1();
  size_[1] = weights.size2();
  
  weights_.attach(weights);  
  
  weights_der_.resize(size_[0], size_[1]);
  weights_der2_.resize(size_[0], size_[1]);
  weights_der_prev_.resize(size_[0], size_[1]);
  weights_learn_coefs_.resize(size_[0], size_[1]);  
  
  weights_der_.assign(0);
  weights_der2_.assign(0);
  weights_der_prev_.assign(0);
  weights_learn_coefs_.assign(1);  
}

void Weights::Attach(Weights &weights, size_t offset, size_t size1, size_t size2, bool order) {
  size_.resize(2);
  size_[0] = size1; size_[1] = size2;
  size_t numel = size1 * size2;
  
  weights_.attach(weights.weights_, offset, size1, size2, order);
  weights_der_.attach(weights.weights_der_, offset, size1, size2, order);
  weights_der2_.attach(weights.weights_der2_, offset, size1, size2, order);
  weights_der_prev_.attach(weights.weights_der_prev_, offset, size1, size2, order);
  weights_learn_coefs_.attach(weights.weights_learn_coefs_, offset, size1, size2, order);  
  
  weights_.reorder(kDefaultOrder, true);
  weights_der_.reorder(kDefaultOrder, false);
  weights_der2_.reorder(kDefaultOrder, false);
  weights_der_prev_.reorder(kDefaultOrder, false);
  weights_learn_coefs_.reorder(kDefaultOrder, false);
}

void Weights::Update(const Params &params, size_t epoch, bool isafter) {
  
  ftype alpha, beta, momentum;
  if (params.alpha_.size() == 1) {
    alpha = params.alpha_[0];
  } else {
    alpha = params.alpha_[epoch];
  }
  if (params.beta_.size() == 1) {
    beta = params.beta_[0];
  } else {
    beta = params.beta_[epoch];
  }
  if (params.momentum_.size() == 1) {
    momentum = params.momentum_[0];
  } else {
    momentum = params.momentum_[epoch];
  }
  if (!isafter) {
    if (momentum == 0) return;
    weights_der_ = weights_der_prev_;    
    weights_der_ *= momentum; 
  } else {
    weights_der_ *= alpha;
    if (beta > 0) {      
      weights_der2_ *= beta;
      weights_der_ += weights_der2_;      
    }
    if (params.adjustrate_ > 0) {      
      Mat signs = weights_der_prev_;      
      signs *= weights_der_;
      weights_learn_coefs_.CondAdd(signs, true, 0, params.adjustrate_);
      weights_learn_coefs_.CondMult(signs, false, 0, 1 - params.adjustrate_);
      weights_learn_coefs_.CondAssign(weights_learn_coefs_, true, params.maxcoef_, params.maxcoef_);
      weights_learn_coefs_.CondAssign(weights_learn_coefs_, false, params.mincoef_, params.mincoef_);      
      weights_der_ *= weights_learn_coefs_;      
    }
    weights_der_prev_ = weights_der_;    
    if (momentum > 0) {
      weights_der_ *= (1 - momentum);    
    }
  }
  // direction that decreases the error
  weights_ -= weights_der_;
  weights_.Validate();  
}

void Weights::Clear() {
  weights_.clear();
  weights_der_.clear();
  weights_der2_.clear();
  weights_der_prev_.clear();
  weights_learn_coefs_.clear(); 
}
