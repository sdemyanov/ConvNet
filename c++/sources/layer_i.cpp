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

#include "layer_i.h"

LayerInput::LayerInput() {
  type_ = "i";
  is_weights_ = false;
  numdim_ = 2;
  batchsize_ = 0;
  length_prev_ = 0;
  norm_ = 0;
  mean_ = 0;
  maxdev_ = 1;
  is_mean_ = false;
  is_maxdev_ = false;
}  
  
void LayerInput::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer == NULL, "The 'i' type layer must be the first one");
  outputmaps_ = 1;
  if (mexIsField(mx_layer, "outputmaps")) {
    ftype outputmaps = mexGetScalar(mexGetField(mx_layer, "outputmaps"));
    mexAssert(1 <= outputmaps, "Outputmaps on the 'i' layer must be greater or equal to 1");
    outputmaps_ = (size_t) outputmaps;    
  }
  mexAssert(mexIsField(mx_layer, "mapsize"), "The first layer must contain the 'mapsize' field");
  std::vector<ftype> mapsize = mexGetVector(mexGetField(mx_layer, "mapsize"));  
  mexAssert(mapsize.size() == numdim_, "Input mapsize length must be 2");  
  mapsize_.resize(numdim_);  
  length_ = outputmaps_;
  size_t numel = 1;
  for (size_t i = 0; i < numdim_; ++i) {    
    mexAssert(1 <= mapsize[i], "Mapsize on the 'i' layer must be greater or equal to 1");
    mapsize_[i] = (size_t) mapsize[i];
    length_ *= mapsize_[i];
    numel *= mapsize_[i];
  }
  if (mexIsField(mx_layer, "norm")) {
    norm_ = mexGetScalar(mexGetField(mx_layer, "norm"));
    mexAssert(norm_ > 0, "Norm on the 'i' layer must be positive");        
  }
  if (mexIsField(mx_layer, "mean")) {
    is_mean_ = true;
    mean_ = mexGetScalar(mexGetField(mx_layer, "mean"));    
  }
  if (mexIsField(mx_layer, "maxdev")) {
    is_maxdev_ = true;
    maxdev_ = mexGetScalar(mexGetField(mx_layer, "maxdev"));    
    mexAssert(maxdev_ > 0, "Maxdev on the 'i' layer must be positive");        
  }
}

void LayerInput::Forward(Layer *prev_layer, int passnum) {  
  batchsize_ = activ_mat_.size1();
  InitMaps(activ_mat_, mapsize_, activ_);
  if (norm_ > 0) {
    activ_mat_.Normalize(norm_);
  }
  if (is_mean_) {
    activ_mat_.AddVect(mean_weights_.get(), 1);
  }
  if (is_maxdev_) {
    activ_mat_.MultVect(maxdev_weights_.get(), 1);
  }  
  activ_mat_.Validate();  
}

void LayerInput::InitWeights(Weights &weights, size_t &offset, bool isgen) {  
  std::vector<size_t> weightssize(2);
  weightssize[0] = 1; weightssize[1] = length_;
  if (is_mean_) {
    mean_weights_.Attach(weights, weightssize, offset);
    offset += length_;  
    if (isgen) mean_weights_.get().assign(0);
  }
  if (is_maxdev_) {
    maxdev_weights_.Attach(weights, weightssize, offset);
    offset += length_;  
    if (isgen) maxdev_weights_.get().assign(1);
  }  
}

size_t LayerInput::NumWeights() const {
  size_t num_weights = 0;
  if (is_mean_) num_weights += length_;
  if (is_maxdev_) num_weights += length_;
  return num_weights;
}