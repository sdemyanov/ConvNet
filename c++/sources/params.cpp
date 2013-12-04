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

#include "params.h"

Params::Params() {
  batchsize_ = 50;
  numepochs_ = 1;
  balance_ = false;
  alpha_ = 1;
  momentum_ = 0;
  maxcoef_ = 1;
  mincoef_ = 1;
}
  
void Params::Init(const mxArray *mx_params) {
  
  mexAssert(mx_params != NULL, "Params array is NULL or empty");
  
  if (mexIsField(mx_params, "batchsize")) {    
    batchsize_ = (size_t) mexGetScalar(mexGetField(mx_params, "batchsize"));    
  }  
  mexPrintMsg("batchsize", batchsize_);
  
  if (mexIsField(mx_params, "numepochs")) {    
    numepochs_ = (size_t) mexGetScalar(mexGetField(mx_params, "numepochs"));    
  }
  mexPrintMsg("numepochs", numepochs_);
  
  if (mexIsField(mx_params, "alpha")) {    
    alpha_ = (float) mexGetScalar(mexGetField(mx_params, "alpha"));    
  }
  mexPrintMsg("alpha", alpha_);
  
  if (mexIsField(mx_params, "momentum")) {    
    momentum_ = (float) mexGetScalar(mexGetField(mx_params, "momentum"));    
  }
  mexPrintMsg("momentum", momentum_);
  
  if (mexIsField(mx_params, "adjustrate")) {    
    adjustrate_ = (float) mexGetScalar(mexGetField(mx_params, "adjustrate"));    
  }
  mexPrintMsg("adjustrate", adjustrate_);
  
  if (mexIsField(mx_params, "maxcoef")) {    
    maxcoef_ = (float) mexGetScalar(mexGetField(mx_params, "maxcoef"));
    mincoef_ = (float) 1 / maxcoef_;    
  }  
  mexPrintMsg("maxcoef", maxcoef_);
  
  if (mexIsField(mx_params, "balance")) {    
    balance_ = (bool) mexGetScalar(mexGetField(mx_params, "balance"));    
  }  
  mexPrintMsg("balance", balance_);
}
