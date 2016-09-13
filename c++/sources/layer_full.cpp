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

#include "layer_full.h"

LayerFull::LayerFull() {
  dims_[2] = 1;
  dims_[3] = 1;
  filters_.dims(2) = 1;
  filters_.dims(3) = 1;
}

void LayerFull::Init(const mxArray *mx_layer, const Layer *prev_layer) {
  mexAssertMsg(mexIsField(mx_layer, "channels"), "The 'full' type layer must contain the 'channels' field");
  filters_.dims(1) = prev_layer->length();
}

void LayerFull::TransformForward(Layer *prev_layer, PassNum passnum) {
  Prod(prev_layer->activ_mat_, false, filters_.get(), true, activ_mat_);
  activ_mat_.Validate();
}

void LayerFull::TransformBackward(Layer *prev_layer) {
  Prod(deriv_mat_, false, filters_.get(), false, prev_layer->deriv_mat_);
  prev_layer->deriv_mat_.Validate();
}

void LayerFull::WeightGrads(Layer *prev_layer, GradInd gradind) {
  if (gradind == GradInd::First) {
    Prod(deriv_mat_, true, prev_layer->activ_mat_, false, filters_.der());
    (filters_.der() *= (lr_coef_ / dims_[0])).Validate();
  } else if (gradind == GradInd::Second) {
    Prod(deriv_mat_, true, prev_layer->activ_mat_, false, filters_.der2());
    (filters_.der2() *= (lr_coef_ / dims_[0])).Validate();
  } else {
    mexAssertMsg(false, "Wrong gradind for WeightGrads");
  }
}
