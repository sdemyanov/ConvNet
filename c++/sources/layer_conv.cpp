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

#include "layer_conv.h"

LayerConv::LayerConv() {
  conv_mode_ = CUDNN_CROSS_CORRELATION;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
}

LayerConv::~LayerConv() {
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void LayerConv::Init(const mxArray *mx_layer, const Layer *prev_layer) {
  mexAssertMsg(mexIsField(mx_layer, "channels"), "The 'conv' type layer must contain the 'channels' field");
  mexAssertMsg(mexIsField(mx_layer, "filtersize"), "The 'conv' type layer must contain the 'filtersize' field");
  filters_.dims(1) = prev_layer->dims_[1];
  Pair upscale = {1, 1};
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc_, padding_[0], padding_[1], stride_[0], stride_[1],
    upscale[0], upscale[1], conv_mode_
  ));
  for (size_t i = 0; i < 2; ++i) {
    dims_[i+2] = 1 + (prev_layer->dims_[i+2] + 2*padding_[i] - filters_.dims(i+2)) / stride_[i];
    mexAssertMsg(1 <= dims_[i+2], "Mapsize on the 'conv' layer must be >= 1");
  }
}

void LayerConv::TransformForward(Layer *prev_layer, PassNum passnum) {
  ConvolutionForward(prev_layer->activ_mat_, filters_.get(),
                     activ_mat_, conv_desc_);
  activ_mat_.Validate();
}

void LayerConv::TransformBackward(Layer *prev_layer) {
  ConvolutionBackwardData(deriv_mat_, filters_.get(),
                          prev_layer->deriv_mat_, conv_desc_);
  prev_layer->deriv_mat_.Validate();
}

void LayerConv::WeightGrads(Layer *prev_layer, GradInd gradind) {
  if (gradind == GradInd::First) {
    ConvolutionBackwardFilter(prev_layer->activ_mat_, deriv_mat_,
                              filters_.der(), conv_desc_);
    (filters_.der() *= (lr_coef_ / dims_[0])).Validate();
  } else if (gradind == GradInd::Second) {
    ConvolutionBackwardFilter(prev_layer->activ_mat_, deriv_mat_,
                              filters_.der2(), conv_desc_);
    (filters_.der2() *= (lr_coef_ / dims_[0])).Validate();
  } else {
    mexAssertMsg(false, "Wrong gradind for WeightGrads");
  }
}
