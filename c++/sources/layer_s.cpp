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

#include "layer_s.h"

LayerScal::LayerScal() {
  type_ = "s";
  is_weights_ = false;
  function_ = "mean";   
  batchsize_ = 0;  
}  
  
void LayerScal::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 's' type layer cannot be after 'f' type layer");
  numdim_ = prev_layer->numdim_;
  outputmaps_ = prev_layer->outputmaps_;
  length_prev_ = prev_layer->length_prev_;
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
  if (mexIsField(mx_layer, "function")) {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
    mexAssert(function_ == "max" || function_ == "mean", 
      "Unknown function for the 's' layer");    
  }
}
    
void LayerScal::Forward(Layer *prev_layer, int passnum) {  

  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  InitMaps(activ_mat_, mapsize_, activ_);  
  
  if (function_ == "max" && maxmat_.size() == 0) {
    maxmat_.resize(batchsize_);
    for (int k = 0; k < batchsize_; ++k) {      
      maxmat_[k].resize(outputmaps_);
    }
  }
  
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif  
  for (int k = 0; k < batchsize_; ++k) {      
    for (size_t i = 0; i < outputmaps_; ++i) {
      if (function_ == "mean") {
        MeanScale(prev_layer->activ_[k][i], scale_, stride_, activ_[k][i]);
      } else if (function_ == "max") {
        MaxMat(prev_layer->activ_[k][i], scale_, stride_, maxmat_[k][i]);        
        MaxScale(prev_layer->activ_[k][i], maxmat_[k][i], activ_[k][i]);       
      }      
    }    
  }
  activ_mat_.Validate();
  /*
  for (int i = 0; i < 10; ++i) {
    mexPrintMsg("Scal: prev_layer->activ_[0][0]", prev_layer->activ_[0][0](0, i)); 
  } */
}

void LayerScal::Backward(Layer *prev_layer) {    
  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_);  
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {
      if (function_ == "mean") {
        MeanScaleDer(deriv_[k][i], scale_, stride_, prev_layer->deriv_[k][i]);
      } else if (function_ == "max") {
        MaxScaleDer(deriv_[k][i], maxmat_[k][i], prev_layer->deriv_[k][i]);        
      }      
    }
  }
  prev_layer->deriv_mat_.Validate();
  /*
  mexPrintMsg("Sum: prev_layer->deriv_", prev_layer->deriv_[0][0].Sum()); 
  for (int i = 0; i < 10; ++i) {
    mexPrintMsg("Scal: prev_layer->deriv_[0][0]", prev_layer->deriv_[0][0](0, i)); 
  } */
}
