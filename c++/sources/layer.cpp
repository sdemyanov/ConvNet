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

#include "layer.h"

Layer::Layer() {
  dims_ = {0, 0, 0, 0};
  function_ = "relu";
  add_bias_ = true;
  padding_ = {0, 0};
  stride_ = {1, 1};
  init_std_ = 0.01;
  lr_coef_ = 1.0;
  bias_coef_ = 1.0;
  dropout_ = 0;
}

void Layer::InitGeneral(const mxArray *mx_layer) {
  type_ = mexGetString(mexGetField(mx_layer, "type"));
  if (mexIsField(mx_layer, "function")) {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
    mexAssertMsg(function_ == "relu" ||
              function_ == "sigm" ||
              function_ == "soft" ||
              function_ == "none",
              "Unknown function code");
  }
  if (mexIsField(mx_layer, "add_bias")) {
    // actual value if defined
    add_bias_ = (mexGetScalar(mexGetField(mx_layer, "add_bias")) > 0);
  }
  if (mexIsField(mx_layer, "mapsize")) {
    std::vector<ftype> mapsize = mexGetVector(mexGetField(mx_layer, "mapsize"));
    mexAssertMsg(mapsize.size() == 2, "Input mapsize length must be 2");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(1 <= mapsize[i] && mapsize[i] < INT_MAX, "Mapsize must be >= 1");
      dims_[i+2] = (int) mapsize[i];
    }
  }
  if (mexIsField(mx_layer, "channels")) {
    ftype channels = mexGetScalar(mexGetField(mx_layer, "channels"));
    mexAssertMsg(1 <= channels && channels < INT_MAX, "Channels num must be >= 1");
    dims_[1] = (int) channels;
    filters_.dims(0) = dims_[1];
    if (add_bias_) { // using actual value here
      biases_.dims() = {1, dims_[1], 1, 1};
    }
  }
  if (mexIsField(mx_layer, "filtersize")) {
    std::vector<ftype> filtersize = mexGetVector(mexGetField(mx_layer, "filtersize"));
    mexAssertMsg(filtersize.size() == 2, "Filtersize must contain 2 values");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(1 <= filtersize[i] && filtersize[i] < INT_MAX, "Filtersize must be >= 1");
      filters_.dims(i+2) = (int) filtersize[i];
    }
  }
  if (mexIsField(mx_layer, "padding")) {
    std::vector<ftype> padding = mexGetVector(mexGetField(mx_layer, "padding"));
    mexAssertMsg(padding.size() == 2, "Padding vector must have 2 values");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(0 <= padding[i] && padding[i] < INT_MAX, "Padding must be non-negative");
      padding_[i] = (int) padding[i];
    }
  }
  if (mexIsField(mx_layer, "stride")) {
    std::vector<ftype> stride = mexGetVector(mexGetField(mx_layer, "stride"));
    mexAssertMsg(stride.size() == 2, "Stride vector has the wrong length");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(1 <= stride[i] && stride[i] < INT_MAX, "Stride must be >= 1");
      stride_[i] = (int) stride[i];
    }
  }
  if (mexIsField(mx_layer, "init_std")) {
    init_std_ = mexGetScalar(mexGetField(mx_layer, "init_std"));
    mexAssertMsg(0 <= init_std_, "init_std must be non-negative");
  }
  if (mexIsField(mx_layer, "bias_coef")) {
    bias_coef_ = mexGetScalar(mexGetField(mx_layer, "bias_coef"));
    mexAssertMsg(0 <= bias_coef_, "bias_coef must be non-negative");
  }
  if (mexIsField(mx_layer, "lr_coef")) {
    lr_coef_ = mexGetScalar(mexGetField(mx_layer, "lr_coef"));
    mexAssertMsg(0 <= lr_coef_, "lr_coef must be non-negative");
  }
  if (mexIsField(mx_layer, "dropout")) {
    dropout_ = mexGetScalar(mexGetField(mx_layer, "dropout"));
    mexAssertMsg(0 <= dropout_ && dropout_ < 1, "dropout must be in the range [0, 1)");
  }
}

void Layer::ResizeActivMat(size_t batchsize, PassNum passnum) {
  dims_[0] = batchsize;
  if (passnum == PassNum::ForwardLinear) {
    // save activ_mat_ from the first pass for Nonlinear function
    Swap(activ_mat_, first_mat_);
  }
  activ_mat_.resize_tensor(dims_);
}

void Layer::ResizeDerivMat() {
  deriv_mat_.resize_tensor(dims_);
}

void Layer::Nonlinear(PassNum passnum) {
  if (function_ == "relu") {
    if (passnum == PassNum::ForwardTest || passnum == PassNum::Forward) { // test and train forward
      activ_mat_.CondAssign(activ_mat_, false, kEps, 0);
    } else if (passnum == PassNum::Backward) {
      deriv_mat_.CondAssign(activ_mat_, false, kEps, 0);
    } else if (passnum == PassNum::ForwardLinear) {
      activ_mat_.CondAssign(first_mat_, false, kEps, 0);
    }
  } else if (function_ == "soft") {
    if (passnum == PassNum::ForwardTest || passnum == PassNum::Forward) { // test and train forward
      activ_mat_.SoftMax();
    } else if (passnum == PassNum::Backward) {
      deriv_mat_.SoftDer(activ_mat_);
    } else if (passnum == PassNum::ForwardLinear) { // third pass
      // no actions as this is the last layer of the third pass,
      // so the results don't go anywhere anyway
      //activ_mat_.SoftDer(first_mat_);
      //activ_mat_.Validate();
    }
    activ_mat_.Validate();
  } else if (function_ == "sigm") {
    if (passnum == PassNum::ForwardTest || passnum == PassNum::Forward) { // test and train forward
      activ_mat_.Sigmoid();
    } else if (passnum == PassNum::Backward) {
      activ_mat_.SigmDer(first_mat_);
    } else if (passnum == PassNum::ForwardLinear) { // third pass
      activ_mat_.SigmDer(first_mat_);
    }
    activ_mat_.Validate();
  } else if (function_ == "none") {
    return;
  } else {
    mexAssertMsg(false, "Unknown function name in Nonlinear");
  }
}

void Layer::AddBias(PassNum passnum) {
  if (add_bias_ == false) return;
  if (passnum == PassNum::ForwardTest || passnum == PassNum::Forward) {
    activ_mat_.AddTensor(biases_.get());
  }
}

void Layer::BiasGrads(GradInd gradind) {
  if (add_bias_ == false) return;
  if (gradind == GradInd::First) {
    ConvolutionBackwardBias(deriv_mat_, biases_.der());
    (biases_.der() *= (lr_coef_ * bias_coef_ / dims_[0])).Validate();
  } else if (gradind == GradInd::Second) {
    ConvolutionBackwardBias(deriv_mat_, biases_.der2());
    (biases_.der2() *= (lr_coef_ * bias_coef_ / dims_[0])).Validate();
  } else {
    mexAssertMsg(false, "Wrong gradind for WeightGrads");
  }
}

void Layer::DropoutForward(PassNum passnum) {
  if (dropout_ > 0) { // dropout
    if (passnum == PassNum::Forward) {
      dropmat_.resize(dims_[0], length());
      dropmat_.rand();
      dropmat_.CondAssign(dropmat_, false, dropout_, 0);
      dropmat_.CondAssign(dropmat_, true, 0, 1);
      activ_mat_ *= dropmat_;
    } else if (passnum == PassNum::ForwardLinear) {
      activ_mat_ *= dropmat_;
    } else if (passnum == PassNum::ForwardTest) {
      activ_mat_ *= (1 - dropout_);
    }
  }
}

void Layer::DropoutBackward() {
  if (dropout_ > 0) {
    deriv_mat_ *= dropmat_;
  }
}

void Layer::InitWeights(Weights &weights, size_t &offset, bool isgen) {
  filters_.AttachFilters(weights, offset);
  offset += filters_.Num();
  if (isgen) {
    (filters_.get().rand() -= 0.5) *= init_std_;
    //filters_.get().randnorm() *= init_std_;
  }
  if (add_bias_) {
    biases_.AttachBiases(weights, offset);
    offset += biases_.Num();
    if (isgen) {
      biases_.get().assign(0);
    }
  }
}

void Layer::RestoreOrder() {
  filters_.RestoreOrder();
}

size_t Layer::NumWeights() const {
  size_t num = filters_.Num();
  if (add_bias_) {
    num += biases_.Num();
  }
  return num;
}
