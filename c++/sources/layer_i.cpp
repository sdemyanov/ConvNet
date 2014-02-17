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
  numdim_ = 2;
  batchsize_ = 0;
  length_prev_ = 0;
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
    norm_ = mexGetVector(mexGetField(mx_layer, "norm"));
    mexAssert(norm_.size() == outputmaps_, "The length of the norm vector is wrong");
    for (size_t i = 0; i < outputmaps_; ++i) {      
      mexAssert(0 < norm_[i], "Norm on the 'i' layer must be positive");          
    }    
  }
  if (mexIsField(mx_layer, "mean")) {
    std::vector<size_t> data_dim = mexGetDimensions(mexGetField(mx_layer, "mean"));
    mexAssert(numdim_ <= data_dim.size() && data_dim.size() <= numdim_ + 1, 
      "Mean matrix on the 'i' layer has wrong the number of dimensions");
    for (size_t i = 0; i < numdim_; ++i) {    
      mexAssert(data_dim[i] == mapsize_[i], "The size of Mean matrix must coincide with the mapsize");
    }
    if (data_dim.size() == numdim_) {
      mexAssert(outputmaps_ == 1, "Not enough Mean matrices for this outputmaps number");      
    } else {
      mexAssert(outputmaps_ == data_dim[numdim_],
        "The number of Mean matrices must coincide with the outputmaps");        
    }
    mean_.resize(outputmaps_);
    ftype *mean = mexGetPointer(mexGetField(mx_layer, "mean"));
    for (size_t i = 0; i < outputmaps_; ++i) {
      mean_[i].attach(mean, mapsize_);
      mean += numel;
    }    
  }
  if (mexIsField(mx_layer, "stdev")) {
    std::vector<size_t> data_dim = mexGetDimensions(mexGetField(mx_layer, "stdev"));
    mexAssert(numdim_ <= data_dim.size() && data_dim.size() <= numdim_ + 1, 
      "Mean matrix on the 'i' layer has wrong the number of dimensions");    
    for (size_t i = 0; i < numdim_; ++i) {    
      mexAssert(data_dim[i] == mapsize_[i], "The size of Stdev matrix must coincide with the mapsize");
    }
    if (data_dim.size() == numdim_) {
      mexAssert(outputmaps_ == 1, "Not enough Stdev matrices for this outputmaps number");      
    } else {
      mexAssert(outputmaps_ == data_dim[numdim_],
        "The number of Stdev matrices must coincide with the outputmaps");        
    }
    stdev_.resize(outputmaps_);
    ftype *stdev = mexGetPointer(mexGetField(mx_layer, "stdev"));
    for (size_t i = 0; i < outputmaps_; ++i) {
      stdev_[i].attach(stdev, mapsize_);
      stdev += numel;
      Mat cond;
      cond.init(mapsize_, 0);    
      cond.CondAssign(stdev_[i], 0, false, 1);    
      mexAssert(cond.Sum() == 0, "All elements of stdev matrix must be positive");
    }    
  }
}

void LayerInput::Forward(Layer *prev_layer, int passnum) {  
  batchsize_ = activ_mat_.size1();
  InitMaps(activ_mat_, mapsize_, activ_);
  datanorm_.resize(batchsize_, outputmaps_);  
  bool is_norm = (norm_.size() > 0);
  bool is_mean = (mean_.size() > 0);
  bool is_stdev = (stdev_.size() > 0);  
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {
      if (is_norm) {
        if (passnum == 0 || passnum == 1) {
          activ_[k][i].Normalize(norm_[i], datanorm_(k, i));
        } else if (passnum == 3) {
          activ_[k][i] *= (norm_[i] / datanorm_(k, i));
        }
      }
      if (is_mean) {
        if (passnum == 0 || passnum == 1) {
          activ_[k][i] -= mean_[i];
        }
      }
      if (is_stdev) activ_[k][i] /= stdev_[i];      
    }    
  }
  activ_mat_.Validate();  
}

void LayerInput::Backward(Layer *prev_layer) {  
  bool is_norm = (norm_.size() > 0);  
  bool is_stdev = (stdev_.size() > 0);
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {    
      if (is_stdev) deriv_[k][i] /= stdev_[i];      
      if (is_norm) deriv_[k][i] *= (norm_[i] / datanorm_(k, i));
    }    
  }
  deriv_mat_.Validate();  
}