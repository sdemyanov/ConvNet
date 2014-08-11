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

#include "layer_t.h"

LayerTrim::LayerTrim() {
  type_ = "t";    
  is_weights_ = false;
  batchsize_ = 0;  
}  
  
void LayerTrim::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 't' type layer cannot be after 'f' type layer");
  numdim_ = prev_layer->numdim_;
  outputmaps_ = prev_layer->outputmaps_;  
  length_prev_ = prev_layer->length_prev_;
  mexAssert(mexIsField(mx_layer, "mapsize"), "The 't' type layer must contain the 'mapsize' field");
  std::vector<ftype> mapsize = mexGetVector(mexGetField(mx_layer, "mapsize"));  
  mexAssert(mapsize.size() == numdim_, "Length of the trim vector and maps dimensionality must coincide");
  mapsize_.resize(numdim_);
  length_ = outputmaps_;
  for (size_t i = 0; i < numdim_; ++i) {
    mexAssert(mapsize[i] <= prev_layer->mapsize_[i], "In 't' layer new mapsize cannot be larger than the old one");    
    mapsize_[i] = (size_t) mapsize[i];    
    length_ *= mapsize_[i];
  }  
}
    
void LayerTrim::Forward(Layer *prev_layer, int passnum) {
  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  InitMaps(activ_mat_, mapsize_, activ_);  
  coords_.resize(batchsize_);
  for (size_t k = 0; k < batchsize_; ++k) {
    coords_[k].resize(outputmaps_);
    for (size_t i = 0; i < outputmaps_; ++i) {
      coords_[k][i].resize(2);
      MaxTrim(prev_layer->activ_[k][i], coords_[k][i], activ_[k][i]);      
    }    
  }
  activ_mat_.Validate();
}

void LayerTrim::Backward(Layer *prev_layer) {  
  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_); 
  for (size_t k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {
      prev_layer->deriv_[k][i].resize(prev_layer->mapsize_);
      MaxTrimDer(deriv_[k][i], coords_[k][i], prev_layer->deriv_[k][i]);            
    }
  }
  prev_layer->deriv_mat_.Validate();
}
