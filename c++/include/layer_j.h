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

#ifndef _LAYER_J_H_
#define _LAYER_J_H_

#define _USE_MATH_DEFINES
#include "layer.h"

class LayerJitter : public Layer {
  
public:
  LayerJitter();
  ~LayerJitter() {};  
  void Init(const mxArray *mx_layer, Layer *prev_layer);
  void Forward(Layer *prev_layer, int passnum);
  void Backward(Layer *prev_layer);
  void CalcWeights(Layer *prev_layer) {};
  void CalcWeights2(Layer *prev_layer, const std::vector<size_t> &invalid) {};
  void UpdateWeights(const Params &params, size_t epoch, bool isafter) {};
  void SetWeights(ftype *&weights, bool isgen) {};
  size_t NumWeights() const { return 0; };

private:
  std::vector<ftype> shift_;
  std::vector<ftype> scale_;
  std::vector<bool> mirror_;
  ftype angle_;  
  ftype default_; // default value to fill in if the new image is out of the original one
  
};

#endif