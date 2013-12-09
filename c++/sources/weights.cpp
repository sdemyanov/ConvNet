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

#include "weights.h"

void Weights::Init(const std::vector<size_t> &newsize, double coef) {
  weights_.resize(newsize);
  (weights_.Rand() -= 0.5) *= coef;
  weights_der_.init(newsize, 0);
  weights_der_prev_.init(newsize, 0);
  weights_learn_coefs_.init(newsize, 1);
  size_ = newsize;
}

void Weights::Update(const Params &params, bool isafter) {

  Mat signs(size_);
  if (!isafter) {
    weights_der_ = weights_der_prev_;
    weights_der_ *= params.momentum_;    
  } else {
    if (params.adjustrate_ > 0) {      
      signs = weights_der_prev_;
      signs.ElemProd(weights_der_);
      weights_learn_coefs_.CondAdd(signs, 0, true, params.adjustrate_);
      weights_learn_coefs_.CondProd(signs, 0, false, 1-params.adjustrate_);
      weights_learn_coefs_.CondAssign(weights_learn_coefs_, params.maxcoef_, true, params.maxcoef_);
      weights_learn_coefs_.CondAssign(weights_learn_coefs_, params.mincoef_, false, params.mincoef_);
    }
    weights_der_prev_ = weights_der_;
    weights_der_ *= (1 - params.momentum_);    
  }  
  weights_ -= (weights_der_.ElemProd(weights_learn_coefs_) *= params.alpha_);
  // direction that decreases the error    
}

Mat& Weights::get() {
  return weights_;
}

const Mat& Weights::get() const {
  return weights_;
}

std::vector<size_t> Weights::size() const {
  return size_;
}

const double& Weights::get(size_t ind) const {
  return weights_(ind);
}

Mat& Weights::der(){
  return weights_der_;
}

double& Weights::der(size_t ind){
  return weights_der_(ind);
}
