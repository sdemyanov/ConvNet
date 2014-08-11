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

#include "params.h"

Params::Params() {
  batchsize_ = 50;
  numepochs_ = 1;
  alpha_.assign(1, 1);  
  momentum_.assign(1, 0);
  adjustrate_ = 0;
  maxcoef_ = 1;
  mincoef_ = 1;
  balance_ = false;
  shuffle_ = true;
  verbose_ = 2;
  seed_ = 0;
}
  
void Params::Init(const mxArray *mx_params) {
  
  if (mexIsField(mx_params, "batchsize")) {    
    batchsize_ = (size_t) mexGetScalar(mexGetField(mx_params, "batchsize"));    
    mexAssert(batchsize_ > 0, "Batchsize must be positive");
  }
  if (mexIsField(mx_params, "numepochs")) {    
    numepochs_ = (size_t) mexGetScalar(mexGetField(mx_params, "numepochs"));    
    mexAssert(numepochs_ > 0, "Numepochs must be positive");
  }
  if (mexIsField(mx_params, "alpha")) {    
    alpha_ = mexGetVector(mexGetField(mx_params, "alpha"));
    mexAssert(alpha_.size() == 1 || alpha_.size() == numepochs_,
      "Wrong length of the alpha vector");
    for (size_t i = 0; i < alpha_.size(); ++i) {
      mexAssert(alpha_[i] >= 0, "alpha must be nonnegative");
    }
  }
  if (mexIsField(mx_params, "momentum")) {    
    momentum_ = mexGetVector(mexGetField(mx_params, "momentum"));
    mexAssert(momentum_.size() == 1 || momentum_.size() == numepochs_,
      "Wrong length of the momentum vector");
    for (size_t i = 0; i < momentum_.size(); ++i) {
      mexAssert(0 <= momentum_[i] && momentum_[i] < 1, "Momentum is out of range [0, 1)");
    }    
  }
  if (mexIsField(mx_params, "adjustrate")) {    
    adjustrate_ = mexGetScalar(mexGetField(mx_params, "adjustrate"));    
    mexAssert(0 <= adjustrate_, "Adjustrate must be non-negative");
  }
  if (mexIsField(mx_params, "maxcoef")) {    
    maxcoef_ = mexGetScalar(mexGetField(mx_params, "maxcoef"));
    mexAssert(1 <= maxcoef_ , "Maxcoef must be larger or equal to 1");
    mincoef_ = 1 / maxcoef_;    
  }
  if (mexIsField(mx_params, "balance")) {    
    balance_ = (bool) mexGetScalar(mexGetField(mx_params, "balance"));    
  }
  if (mexIsField(mx_params, "shuffle")) {    
    shuffle_ = (bool) mexGetScalar(mexGetField(mx_params, "shuffle"));    
  }
  if (mexIsField(mx_params, "verbose")) {    
    verbose_ = (size_t) mexGetScalar(mexGetField(mx_params, "verbose"));    
  }
  if (mexIsField(mx_params, "seed")) {    
    seed_ = (size_t) mexGetScalar(mexGetField(mx_params, "seed"));    
  }
}
