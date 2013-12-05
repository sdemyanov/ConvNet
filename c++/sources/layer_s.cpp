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

#include "layer_s.h"

LayerScal::LayerScal() {
  type_ = "s";    
}  
  
void LayerScal::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 's' type layer cannot be after 'f' type layer");
  mexAssert(mexIsField(mx_layer, "scale"), "The 's' type layer must contain the 'scale' field");
  mapsize_.resize(prev_layer->mapsize_.size());
  std::vector<double> scale = mexGetVector(mexGetField(mx_layer, "scale"));  
  mexAssert(scale.size() == mapsize_.size(), "Length of scale vector and maps dimensionality must coincide");
  scale_.resize(scale.size());
  for (size_t i = 0; i < mapsize_.size(); ++i) {
    scale_[i] = (size_t) scale[i];
    mexAssert(1 <= scale_[i], "Scale size on the 's' layer must be greater or equal to 1");
    mapsize_[i] = ceil((double) prev_layer->mapsize_[i] / scale_[i]);
  }    
  outputmaps_ = prev_layer->outputmaps_;
  if (!mexIsField(mx_layer, "function")) {
    function_ = "mean";
  } else {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
  }
  std::string errmsg = function_ + " - unknown function for the layer";    
  mexAssert(function_ == "max" || function_ == "mean", errmsg);
  activ_.resize(outputmaps_);
  deriv_.resize(outputmaps_);  
  
}
    
void LayerScal::Forward(const Layer *prev_layer, bool istrain) {
  
  batchsize_ = prev_layer->batchsize_;
  for (size_t i = 0; i < outputmaps_; ++i) {
    activ_[i].resize(batchsize_);
    for (size_t k = 0; k < batchsize_; ++k) {      
      activ_[i][k].resize(mapsize_);
      if (function_ == "mean") {
        prev_layer->activ_[i][k].MeanScale(scale_, activ_[i][k]);
      } else if (function_ == "max") {
        prev_layer->activ_[i][k].MaxScale(scale_, activ_[i][k]);
      } else {
        mexAssert(false, "LayerScal::Forward");
      }
    }    
  }  
}

void LayerScal::Backward(Layer *prev_layer) {
  
  for (size_t i = 0; i < outputmaps_; ++i) {
    prev_layer->deriv_[i].resize(batchsize_);
    for (size_t k = 0; k < batchsize_; ++k) {
      prev_layer->deriv_[i][k].resize(prev_layer->mapsize_);
      if (function_ == "mean") {
        deriv_[i][k].MeanScaleDer(scale_, prev_layer->deriv_[i][k]);
      } else if (function_ == "max") {
        deriv_[i][k].MaxScaleDer(scale_, activ_[i][k], prev_layer->activ_[i][k], prev_layer->deriv_[i][k]);
      } else {
        mexAssert(false, "LayerScal::Forward");
      }      
    }
  }
}
