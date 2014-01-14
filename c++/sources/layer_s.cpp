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
  batchsize_ = 0;  
}  
  
void LayerScal::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 's' type layer cannot be after 'f' type layer");
  numdim_ = prev_layer->numdim_;
  outputmaps_ = prev_layer->outputmaps_;
  length_prev_ = prev_layer->outputmaps_;
  mexAssert(mexIsField(mx_layer, "scale"), "The 's' type layer must contain the 'scale' field");
  std::vector<ftype> scale = mexGetVector(mexGetField(mx_layer, "scale"));  
  mexAssert(scale.size() == numdim_, "Length of the scale vector and maps dimensionality must coincide");
  scale_.resize(numdim_);
  stride_.resize(numdim_);
  mapsize_.resize(numdim_);
  length_ = outputmaps_;
  for (size_t i = 0; i < numdim_; ++i) {
    mexAssert(1 <= scale[i] && scale[i] <= prev_layer->mapsize_[i], "Scale on the 's' layer must be in the range [1, previous_layer_mapsize]");
    scale_[i] = (size_t) scale[i];    
    stride_[i] = scale_[i];
    mapsize_[i] = ceil((ftype) prev_layer->mapsize_[i] / stride_[i]);    
    length_ *= mapsize_[i];
  }
  
  if (mexIsField(mx_layer, "stride")) {
    std::vector<ftype> stride = mexGetVector(mexGetField(mx_layer, "stride"));
    mexAssert(stride.size() == numdim_, "Stride vector has the wrong length");
    length_ = outputmaps_;
    for (size_t i = 0; i < numdim_; ++i) {
      mexAssert(1 <= stride[i] && stride[i] <= prev_layer->mapsize_[i], "Stride on the 's' layer must be in the range [1, previous_layer_mapsize]");    
      stride_[i] = (size_t) stride[i];
      mapsize_[i] = ceil((ftype) prev_layer->mapsize_[i] / stride_[i]);
      length_ *= mapsize_[i];
    }
  }  
  function_ = "mean";   
  if (mexIsField(mx_layer, "function")) {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
  }
  std::string errmsg = function_ + " - unknown function for the layer";    
  mexAssert(function_ == "max" || function_ == "mean", errmsg);
  
}
    
void LayerScal::Forward(Layer *prev_layer, bool istrain) {  

  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  activ_.assign(batchsize_, std::vector<Mat>(outputmaps_));
  InitMaps(activ_mat_, mapsize_, activ_);  
  for (size_t k = 0; k < batchsize_; ++k) {      
    for (size_t i = 0; i < outputmaps_; ++i) {
      if (function_ == "mean") {
        MeanScale(prev_layer->activ_[k][i], scale_, stride_, activ_[k][i]);
      } else if (function_ == "max") {
        MaxScale(prev_layer->activ_[k][i], scale_, stride_, activ_[k][i]);
      } else {
        mexAssert(false, "LayerScal::Forward");
      }      
    }    
  }
  if (!istrain) prev_layer->activ_mat_.clear();  
  /*
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Scal: activ_[0][0]", activ_[0][0](0, i)); 
  } */
}

void LayerScal::Backward(Layer *prev_layer) {  
  if (prev_layer->type_ == "i" || prev_layer->type_ == "j") return;
  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  prev_layer->deriv_.assign(prev_layer->batchsize_, std::vector<Mat>(prev_layer->outputmaps_));
  InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_);  
  for (size_t k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {
      if (function_ == "mean") {
        MeanScaleDer(deriv_[k][i], scale_, stride_, prev_layer->deriv_[k][i]);
      } else if (function_ == "max") {
        MaxScaleDer(deriv_[k][i], activ_[k][i], prev_layer->activ_[k][i], 
                    scale_, stride_, prev_layer->deriv_[k][i]);
      } else {
        mexAssert(false, "LayerScal::Forward");
      }      
    }
  }  
}
