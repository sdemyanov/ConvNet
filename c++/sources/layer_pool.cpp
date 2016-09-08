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

#include "layer_pool.h"

LayerPool::LayerPool() {
  function_ = "none";
  add_bias_ = false;
  pooling_ = "max";
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc_));
}

LayerPool::~LayerPool() {
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc_));
}

void LayerPool::Init(const mxArray *mx_layer, const Layer *prev_layer) {
  dims_[1] = prev_layer->dims_[1];

  if (mexIsField(mx_layer, "pooling")) {
    pooling_ = mexGetString(mexGetField(mx_layer, "pooling"));
    mexAssertMsg(pooling_ == "max" || pooling_ == "avg", "Unknown pooling type");
  }
  mexAssertMsg(mexIsField(mx_layer, "scale"), "The 'pool' type layer must contain the 'scale' field");
  std::vector<ftype> scale = mexGetVector(mexGetField(mx_layer, "scale"));
  mexAssertMsg(scale.size() == 2, "Length of the scale vector and maps dimensionality must coincide");
  for (size_t i = 0; i < 2; ++i) {
    mexAssertMsg(1 <= scale[i] && scale[i] <= prev_layer->dims_[i+2],
     "Scale on the 's' layer must be in the range [1, previous_layer_mapsize]");
    scale_[i] = (int) scale[i];
  }
  mexAssertMsg(mexIsField(mx_layer, "stride"), "The 'pool' type layer must contain the 'stride' field");
  for (size_t i = 0; i < 2; ++i) {
    mexAssertMsg(1 <= stride_[i] && stride_[i] <= prev_layer->dims_[i+2],
      "Stride on the 'pool' layer must be in the range [1, previous_layer_mapsize]");
   }
  // setting CUDNN parameters
  cudnnPoolingMode_t mode;
  if (pooling_ == "max") {
    mode = CUDNN_POOLING_MAX;
  } else if (pooling_ == "avg") {
    mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  }
  cudnnNanPropagation_t nan_prop_mode = CUDNN_PROPAGATE_NAN;
  CUDNN_CALL(cudnnSetPooling2dDescriptor(
    pool_desc_, mode, nan_prop_mode, scale_[0], scale_[1],
    padding_[0], padding_[1], stride_[0], stride_[1]
  ));
  for (size_t i = 0; i < 2; ++i) {
    dims_[i+2] = 1 + (prev_layer->dims_[i+2] + 2*padding_[i] - scale_[i]) / stride_[i];
  }
}

void LayerPool::TransformForward(Layer *prev_layer, PassNum passnum) {
  if (pooling_ == "avg") {
    Pooling(prev_layer->activ_mat_, activ_mat_, pool_desc_);
  } else if (pooling_ == "max") {
    if (passnum == PassNum::ForwardTest || passnum == PassNum::Forward) {
      Pooling(prev_layer->activ_mat_, activ_mat_, pool_desc_);
    } else if (passnum == PassNum::ForwardLinear) {
      // use deriv_mat_'s to identify which locations to propagate to activ_mat_
      PoolingUndo(prev_layer->deriv_mat_, deriv_mat_,
                  activ_mat_, prev_layer->activ_mat_,
                  pool_desc_, false);
    }
  }
}

void LayerPool::TransformBackward(Layer *prev_layer) {
  PoolingUndo(prev_layer->activ_mat_, activ_mat_,
              deriv_mat_, prev_layer->deriv_mat_,
              pool_desc_, true);
}
