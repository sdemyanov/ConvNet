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
  is_weights_ = true;
  batchsize_ = 0;  
  is_dev_ = true;
}  
  
void LayerNorm::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  numdim_ = prev_layer->numdim_;
  outputmaps_ = prev_layer->outputmaps_;
  length_prev_ = prev_layer->length_prev_;  
  length_ = prev_layer->length_;
  mapsize_.resize(numdim_);  
  size_t numel = 1;
  for (size_t i = 0; i < numdim_; ++i) {
    mapsize_[i] = prev_layer->mapsize_[i];    
    numel *= mapsize_[i];
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
    mean_in_.resize(outputmaps_);
    ftype *mean = mexGetPointer(mexGetField(mx_layer, "mean"));
    for (size_t i = 0; i < outputmaps_; ++i) {
      mean_in_[i].attach(mean, mapsize_);
      mean += numel;
    }    
  }
  if (mexIsField(mx_layer, "stdev")) {
    if (mexIsString(mexGetField(mx_layer, "stdev")) && 
        mexGetString(mexGetField(mx_layer, "stdev")) == "no") {
        is_dev_ = false;        
    } else {
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
      stdev_in_.resize(outputmaps_);
      ftype *stdev = mexGetPointer(mexGetField(mx_layer, "stdev"));
      for (size_t i = 0; i < outputmaps_; ++i) {
        stdev_in_[i].attach(stdev, mapsize_);
        stdev += numel;
        Mat cond;
        cond.init(mapsize_, 0);    
        cond.CondAssign(stdev_in_[i], 0, false, 1);    
        mexAssert(cond.Sum() == 0, "All elements of stdev matrix must be positive");
      }    
    }
  }
  if (mexIsField(mx_layer, "filter")) {
    std::vector<size_t> labels_dim = mexGetDimensions(mexGetField(mx_layer, "filter"));  
    mexAssert(labels_dim.size() == 2, "The label array must have 2 dimensions");
    filter_ = mexGetMatrix(mexGetField(mx_layer, "filter"));      
    mexAssert(filter_.size1() % 2 == 1, "Filter must have odd number of rows");
    mexAssert(filter_.size2() % 2 == 1, "Filter must have odd number of columns");
  }
}

void LayerNorm::Forward(Layer *prev_layer, int passnum) {  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  InitMaps(activ_mat_, mapsize_, activ_);  
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {
      activ_[k][i] = prev_layer->activ_[k][i];
      activ_[k][i] += mean_[i].get();      
      if (is_dev_) activ_[k][i] *= stdev_[i].get();      
    }    
  }
  activ_mat_.Validate();   
}

void LayerNorm::Backward(Layer *prev_layer) {  
  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_);   
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {
    for (size_t i = 0; i < outputmaps_; ++i) {    
      prev_layer->deriv_[k][i] = deriv_[k][i];
      if (is_dev_) prev_layer->deriv_[k][i] *= stdev_[i].get();      
    }    
  }
  prev_layer->deriv_mat_.Validate();    
}

void LayerNorm::CalcWeights(Layer *prev_layer) {  
  
  for (size_t i = 0; i < outputmaps_; ++i) {
    mean_[i].der().assign(0);
    if (is_dev_) stdev_[i].der().assign(0);
  }
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {    
    for (size_t i = 0; i < outputmaps_; ++i) {  
      Mat meander = deriv_[k][i];
      if (is_dev_) meander *= stdev_[i].get();
      #if USE_MULTITHREAD == 1
        #pragma omp critical
      #endif
      mean_[i].der() += meander;
      
      if (is_dev_) {
        Mat stdevder = prev_layer->activ_[k][i];      
        (stdevder += mean_[i].get()) *= deriv_[k][i];
        #if USE_MULTITHREAD == 1
          #pragma omp critical
        #endif
        stdev_[i].der() += stdevder;
      }
    }
  }
  for (size_t i = 0; i < outputmaps_; ++i) {
    mean_[i].der() /= batchsize_;
    mean_[i].der().Validate();
    if (is_dev_) {
      stdev_[i].der() /= batchsize_;
      stdev_[i].der().Validate();
    }
    if (!filter_.isempty()) {
      std::vector<size_t> padding(numdim_);
      padding[0] = (filter_.size1() - 1) / 2;
      padding[1] = (filter_.size2() - 1) / 2;    
      Mat filt_mat(mapsize_);
      Filter(mean_[i].der(), filter_, padding, filt_mat);
      mean_[i].der() = filt_mat;
      if (is_dev_) {
        Filter(stdev_[i].der(), filter_, padding, filt_mat);
        stdev_[i].der() = filt_mat;
      }
    }
  }  
}

void LayerNorm::InitWeights(Weights &weights, size_t &offset, bool isgen) {
  size_t numel = 1;
  for (size_t i = 0; i < numdim_; ++i) {
    numel *= mapsize_[i];
  }    
  mean_.resize(outputmaps_);
  for (size_t i = 0; i < outputmaps_; ++i) {    
    mean_[i].Attach(weights, mapsize_, offset);    
    offset += numel;
    if (isgen) {
      mean_[i].get().assign(0);
      if (mean_in_.size() > 0) {        
        mean_[i].get() -= mean_in_[i];                
      }      
    }
  }
  if (is_dev_) {
    stdev_.resize(outputmaps_);
    for (size_t i = 0; i < outputmaps_; ++i) {    
      stdev_[i].Attach(weights, mapsize_, offset);
      offset += numel;
      if (isgen) {
        stdev_[i].get().assign(1);
        if (stdev_in_.size() > 0) {
          stdev_[i].get() /= stdev_in_[i];
        }      
      }
    }  
  }
}

void LayerNorm::UpdateWeights(const Params &params, size_t epoch, bool isafter) {
  for (size_t i = 0; i < outputmaps_; ++i) {    
    mean_[i].Update(params, epoch, isafter);
    if (is_dev_) stdev_[i].Update(params, epoch, isafter);
  }    
}

size_t LayerNorm::NumWeights() const {
  size_t num_weights = outputmaps_;
  for (size_t i = 0; i < numdim_; ++i) {
    num_weights *= mapsize_[i];
  }
  if (is_dev_) num_weights *= 2;
  return num_weights; // *2 because mean and stdev  
}
