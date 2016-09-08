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

#ifndef _NET_H_
#define _NET_H_

#include "layer.h"
#include "params.h"
#include "layer_input.h"
#include "layer_jitt.h"
#include "layer_conv.h"
#include "layer_deconv.h"
#include "layer_pool.h"
#include "layer_full.h"

class Net {

private:
  std::vector<Layer*> layers_;
  size_t first_layer_, first_trained_;
  Weights weights_;
  Params params_;
  MatCPU data_, labels_, preds_;
  MatCPU trainerrors_;
  //MatGPU classcoefs_; // in fact vector
  MatGPU lossmat_, lossmat2_;

  void ReadData(const mxArray *mx_data);
  void ReadLabels(const mxArray *mx_labels);
  void InitActiv(const MatGPU &data_batch);
  void Forward(MatGPU &pred, PassNum passnum, GradInd gradind);
  void InitDeriv(const MatGPU &labels_batch, ftype &loss);
  void InitActiv2(ftype &loss, int normfun);
  void InitActiv3(ftype coef, int normfun);
  void Backward(PassNum passnum, GradInd gradind);
  void UpdateWeights();

public:
  Net(const mxArray *mx_params);
  ~Net();
  void InitLayers(const mxArray *mx_layers);
  void Train(const mxArray *mx_data, const mxArray *mx_labels);
  void Classify(const mxArray *mx_data, const mxArray *mx_labels, mxArray *&mx_pred);
  void InitWeights(const mxArray *mx_weights_in);
  void GetErrors(mxArray *&mx_errors) const;
  void GetWeights(mxArray *&mx_weights) const;
  size_t NumWeights() const;

};

#endif
