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

void Weights::Init(ftype *weights, const std::vector<size_t> &newsize, ftype coef) {
  Init(weights, newsize);
  (weights_.rand() -= 0.5) *= coef;  
}

void Weights::Init(ftype *weights, const std::vector<size_t> &newsize) {
  size_ = newsize;
  weights_.attach(weights, size_);  
  weights_der_.init(size_, 0);
  weights_der2_.init(size_, 0);
  weights_der_prev_.init(size_, 0);
  weights_learn_coefs_.init(size_, 1);  
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
    weights_der2_ *= beta;
    weights_der_ += weights_der2_;    
    if (params.adjustrate_ > 0) {      
      Mat signs = weights_der_prev_;      
      signs *= weights_der_;
      weights_learn_coefs_.CondAdd(signs, 0, true, params.adjustrate_);
      weights_learn_coefs_.CondProd(signs, 0, false, 1-params.adjustrate_);
      weights_learn_coefs_.CondAssign(weights_learn_coefs_, params.maxcoef_, true, params.maxcoef_);
      weights_learn_coefs_.CondAssign(weights_learn_coefs_, params.mincoef_, false, params.mincoef_);      
      weights_der_ *= weights_learn_coefs_;      
    }
    weights_der_prev_ = weights_der_;    
    if (momentum > 0) {
      weights_der_ *= (1 - momentum);    
    }
  }  
  weights_ -= weights_der_;
  weights_.Validate();
  // direction that decreases the error
}
