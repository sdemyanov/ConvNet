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

#ifndef _LAYER_C_H_
#define _LAYER_C_H_

#include "layer.h"

class LayerConv : public Layer {
  
public:  
  LayerConv();  
  ~LayerConv() {};
  void Init(const mxArray *mx_layer, Layer *prev_layer);
  void Forward(Layer *prev_layer, bool istrain);
  void Backward(Layer *prev_layer);
  void UpdateWeights(const Params &params, size_t epoch, bool isafter);
  void GetWeights(ftype *&weights, ftype *weights_end) const;
  void SetWeights(ftype *&weights, ftype *weights_end);
  size_t NumWeights() const;
  
private:
  std::vector<size_t> kernelsize_;  
  std::vector< std::vector<Weights> > kernels_;
  Weights biases_;
  std::vector<size_t> padding_;
  
};

#endif