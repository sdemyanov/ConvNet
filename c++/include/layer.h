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

#ifndef _LAYER_H_
#define _LAYER_H_

#include "mat.h"
#include "weights.h"
#include "params.h"
#include "mex_util.h"

class Layer {
  
public:
  Mat activ_mat_, deriv_mat_; // activations, derivatives
  std::vector< std::vector<Mat> > activ_, deriv_;  // same, but split on the maps if necessary
  std::vector<size_t> mapsize_;
  size_t numdim_, outputmaps_, batchsize_, length_, length_prev_;  
  // length_ - total number of values on the layer in one sample
  // length_prev_ - meaning depends on the layer
  bool is_weights_;  
  std::string type_, function_;  
  
  Layer() {};
  virtual ~Layer() {};
  virtual void Init(const mxArray *mx_layer, Layer *prev_layer) = 0;
  virtual void Forward(Layer *prev_layer, int passnum) = 0;
  virtual void Backward(Layer *prev_layer) = 0;
  virtual void InitWeights(Weights &weights, size_t &offset, bool isgen) = 0;
  virtual void CalcWeights(Layer *prev_layer) = 0;
  virtual void UpdateWeights(const Params &params, size_t epoch, bool isafter) = 0;
  virtual size_t NumWeights() const = 0;  

  void Nonlinear(int passnum);  
  
};

#endif