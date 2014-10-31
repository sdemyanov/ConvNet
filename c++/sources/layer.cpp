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

#include "layer.h"

void Layer::Nonlinear(int passnum) {
  if (passnum == 0 || passnum == 1) { // test and train forward
    if (function_ == "soft") {
      activ_mat_.SoftMax();
    } else if (function_ == "sigm") {
      activ_mat_.Sigmoid();
    } else if (function_ == "relu") {
      activ_mat_.CondAssign(activ_mat_, false, 0, 0);
    } else if (function_ == "SVM") {    
    }
    activ_mat_.Validate();
  } else if (passnum == 2) { // backward
    if (function_ == "soft") {
      deriv_mat_.SoftDer(activ_mat_);
    } else if (function_ == "sigm") {
      deriv_mat_.SigmDer(activ_mat_);
    } else if (function_ == "relu") {        
      deriv_mat_.CondAssign(activ_mat_, false, 0, 0);
    } else if (function_ == "SVM") {    
    }      
    deriv_mat_.Validate();  
  } else if (passnum == 3) { // third pass
    if (function_ == "soft") {
      mexAssert(false, "There should not be 'softmax' layer on the 3rd pass");      
    } else if (function_ == "sigm") {
      activ_mat_.SigmDer(deriv_mat_);
    } else if (function_ == "relu") {
      activ_mat_.CondAssign(deriv_mat_, false, 0, 0);
    } else if (function_ == "SVM") {    
    }
    activ_mat_.Validate();
  }  
}
