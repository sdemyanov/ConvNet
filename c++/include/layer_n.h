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

#ifndef _LAYER_N_H_
#define _LAYER_N_H_

#include "layer.h"

class LayerNorm : public Layer {
  
public:
  LayerNorm();
  ~LayerNorm() {};  
  void Init(const mxArray *mx_layer, Layer *prev_layer);
  void Forward(Layer *prev_layer, int passnum);
  void Backward(Layer *prev_layer);
  void CalcWeights(Layer *prev_layer);
  void InitWeights(Weights &weights, size_t &offset, bool isgen);
  void UpdateWeights(const Params &params, size_t epoch, bool isafter);
  size_t NumWeights() const;
  
private:
  std::vector<Mat> mean_in_;
  std::vector<Mat> stdev_in_;
  std::vector<Weights> mean_;
  std::vector<Weights> stdev_;
  Mat filter_;
  bool is_dev_;
  
};

#endif