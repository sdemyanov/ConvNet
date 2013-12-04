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

#ifndef _WEIGHTS_H_
#define _WEIGHTS_H_

#include "mat.h"
#include "params.h"

class Weights {
public:
  void Init(const std::vector<size_t> &newsize, double coef);
  void Update(const Params &params, bool isafter);    
  Mat& get();
  const Mat& get() const;
  const double& get(size_t ind) const;
  Mat& der();  
  double& der(size_t ind);  
  
private:
  Mat weights_;
  Mat weights_der_;
  Mat weights_der_prev_;
  Mat weights_learn_coefs_; 
  
};

#endif