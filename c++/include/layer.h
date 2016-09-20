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

#ifndef _LAYER_H_
#define _LAYER_H_

#include "mat_gpu.h"
#include "weights.h"
#include "params.h"
#include "mex_util.h"

// ForwardLinear multiplies on the same matrix of gradients as the backward pass.
// It uses activ_mat from the first pass stored in first_mat.
// BackwardLinear is the same as Backward, but uses first_mat_ for nonlinearities.
enum class PassNum {ForwardTest, Forward, Backward, ForwardLinear, BackwardLinear};

// index of where the gradients are stored
enum class GradInd {Nowhere, First, Second};

class Layer {

public:
  // activations, derivatives, first activations
  MatGPU activ_mat_, deriv_mat_, first_mat_;
  // batchsize, channels, height, width
  Dim dims_;
  std::string type_, function_;
  ftype lr_coef_;

  inline size_t length() const { return dims_[1] * dims_[2] * dims_[3]; }

  Layer();
  virtual ~Layer() {};
  virtual void Init(const mxArray *mx_layer, const Layer *prev_layer) = 0;
  virtual void TransformForward(Layer *prev_layer, PassNum passnum) = 0;
  virtual void TransformBackward(Layer *prev_layer) = 0;
  virtual void WeightGrads(Layer *prev_layer, GradInd gradind) = 0;

  void InitGeneral(const mxArray *mx_layer);
  void InitWeights(Weights &weights, size_t &offset, bool isgen);
  void ResizeActivMat(size_t batchsize, PassNum passnum);
  void ResizeDerivMat();
  void AddBias(PassNum passnum);
  void BiasGrads(PassNum passnum, GradInd gradind);
  void DropoutForward(PassNum passnum);
  void DropoutBackward();
  void UpdateWeights(const Params &params);
  void RestoreOrder();
  void Nonlinear(PassNum passnum);
  size_t NumWeights() const;

  Weights filters_, biases_;
  bool add_bias_;

protected:
  Pair padding_, stride_;
  ftype init_std_, bias_coef_, dropout_;
  MatGPU dropmat_;

};

#endif
