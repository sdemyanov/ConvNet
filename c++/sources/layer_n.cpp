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

#include "layer_n.h"

LayerNorm::LayerNorm() {
  type_ = "n";
  function_ = "none";
  normsize_ = 1;
  scale_ = 0;
  pow_ = 1;
  batchsize_ = 0;  
}  
  
void LayerNorm::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 'n' type layer cannot be after 'f' type layer");
  numdim_ = prev_layer->numdim_;
  mapsize_.resize(numdim_);
  for (size_t i = 0; i < numdim_; ++i) {
    mapsize_[i] = prev_layer->mapsize_[i];      
  }  
  outputmaps_ = prev_layer->outputmaps_;
  length_ = prev_layer->length_;
  length_prev_ = prev_layer->length_prev_; // not used
  mexAssert(mexIsField(mx_layer, "normsize"), "The 's' type layer must contain the 'normsize' field");
  ftype normsize = mexGetScalar(mexGetField(mx_layer, "normsize"));
  mexAssert(1 <= normsize && normsize <= outputmaps_, "Normsize cannot be lower than 1 and larger than outputmaps");
  normsize_ = (size_t) normsize;
  scale_ = mexGetScalar(mexGetField(mx_layer, "scale"));
  pow_ = mexGetScalar(mexGetField(mx_layer, "pow"));
}
    
void LayerNorm::Forward(Layer *prev_layer, int passnum) {  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  #if COMP_REGIME != 2 // CPU
    mexAssert("Norm layer is not implemented on CPU");
  #else // GPU  
    LocalResponseNorm(prev_layer->activ_mat_, activ_mat_, 
                      prev_layer->mapsize_, normsize_, scale_, pow_);
  #endif
}

void LayerNorm::Backward(Layer *prev_layer) {    
  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  #if COMP_REGIME != 2
    mexAssert("Norm layer is not implemented on CPU");
  #else // GPU    
    LocalResponseNormUndo(prev_layer->activ_mat_, activ_mat_,
                          deriv_mat_, prev_layer->deriv_mat_,
                          prev_layer->mapsize_, normsize_, scale_, pow_);
  #endif
}
