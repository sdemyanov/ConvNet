/*
Copyright (C) 2016 Sergey Demyanov.
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

#ifndef _LAYER_JITT_H_
#define _LAYER_JITT_H_

#include "layer.h"

class LayerJitt : public Layer {

public:
  LayerJitt();
  ~LayerJitt() {};
  void Init(const mxArray *mx_layer, const Layer *prev_layer);
  void TransformForward(Layer *prev_layer, PassNum passnum);
  void TransformBackward(Layer *prev_layer);
  void WeightGrads(Layer *prev_layer, GradInd gradind) {};

private:
  // maximal values of distortion
  MatGPU shift_, scale_, mirror_;
  // default value to fill in if the new image is out of the original one
  ftype angle_, defval_;
  // actual values of distorion for all batch images
  MatGPU shift_mat_, scale_mat_, mirror_mat_, angle_mat_;

  MatGPU eigenvectors_;
  ftype noise_std_;
  bool randtest_;



};

#endif
