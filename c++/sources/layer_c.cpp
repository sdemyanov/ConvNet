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

#include "layer_c.h"
#include <ctime>

LayerConv::LayerConv() {
  type_ = "c";
  batchsize_ = 0;  
}
  
void LayerConv::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 'c' type layer cannot be after 'f' type layer");
  numdim_ = prev_layer->numdim_;
  length_prev_ = prev_layer->outputmaps_;
  mexAssert(mexIsField(mx_layer, "outputmaps"), "The 'c' type layer must contain the 'outputmaps' field");
  ftype outputmaps = mexGetScalar(mexGetField(mx_layer, "outputmaps"));
  mexAssert(1 <= outputmaps, "Outputmaps on the 'i' layer must be greater or equal to 1");
  outputmaps_ = (size_t) outputmaps;
  function_ = "sigmoid";  
  if (mexIsField(mx_layer, "function")) {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
  }
  mexAssert(mexIsField(mx_layer, "kernelsize"), "The 'c' type layer must contain the 'kernelsize' field");
  std::vector<ftype> kernelsize = mexGetVector(mexGetField(mx_layer, "kernelsize"));
  mexAssert(kernelsize.size() == numdim_, "Kernels and maps must be the same dimensionality");
  kernelsize_.resize(numdim_);    
  for (size_t i = 0; i < numdim_; ++i) {
    mexAssert(1 <= kernelsize[i], "Kernelsize on the 'c' layer must be greater or equal to 1");
    kernelsize_[i] = (size_t) kernelsize[i];    
  }
  padding_.assign(numdim_, 0);
  if (mexIsField(mx_layer, "padding")) {
    std::vector<ftype> padding = mexGetVector(mexGetField(mx_layer, "padding"));
    mexAssert(padding.size() == numdim_, "Padding vector has the wrong length");
    for (size_t i = 0; i < numdim_; ++i) {
      mexAssert(0 <= padding[i] && padding[i] <= kernelsize_[i] - 1, "Padding on the 'c' layer must be in the range [0, kernelsize-1]");
      padding_[i] = (size_t) padding[i];
    }
  }
  mapsize_.resize(numdim_);
  length_ = outputmaps_;
  for (size_t i = 0; i < numdim_; ++i) {
    mapsize_[i] = prev_layer->mapsize_[i] + 2*padding_[i] - kernelsize_[i] + 1;    
    mexAssert(1 <= mapsize_[i], "Mapsize on the 'c' layer must be greater than 1");
    length_ *= mapsize_[i];
  }  
  std::string errmsg = function_ + " - unknown function for the layer";      
  mexAssert(function_ == "sigmoid" || function_ == "relu", errmsg);    
}
    
void LayerConv::Forward(Layer *prev_layer, bool istrain) {
  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  activ_.assign(batchsize_, std::vector<Mat>(outputmaps_));
  InitMaps(activ_mat_, mapsize_, activ_);
  
  for (size_t k = 0; k < batchsize_; ++k) {  
    for (size_t i = 0; i < outputmaps_; ++i) {
      activ_[k][i].assign(biases_.get(i));
      for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {
        Mat act_mat(mapsize_);
        Filter(prev_layer->activ_[k][j], kernels_[i][j].get(), padding_, act_mat);
        activ_[k][i] += act_mat;        
      }
      if (function_ == "sigmoid") {
        activ_[k][i].Sigmoid();
      } else if (function_ == "relu") {
        activ_[k][i].ElemMax(0);
      } else {
        mexAssert(false, "LayerConv::Forward");
      }    
    }    
  }
  if (!istrain) prev_layer->activ_mat_.clear();  
  /*
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Conv: activ_", activ_[1][1](0, i)); 
  }*/
    
}

void LayerConv::Backward(Layer *prev_layer) {

  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  prev_layer->deriv_.assign(prev_layer->batchsize_, std::vector<Mat>(prev_layer->outputmaps_));
  InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_); 

  std::vector<size_t> padding_der(numdim_);
  for (size_t i = 0; i < numdim_; ++i) {
    padding_der[i] = kernelsize_[i] - 1 - padding_[i];
  }
  biases_.der().assign(0);
  for (size_t i = 0; i < outputmaps_; ++i) {    
    for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {    
      kernels_[i][j].der().assign(0);      
    }
  }
  for (size_t k = 0; k < batchsize_; ++k) {
    for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {    
      prev_layer->deriv_[k][j].init(prev_layer->mapsize_, 0);      
    }
    for (size_t i = 0; i < outputmaps_; ++i) {
      if (function_ == "sigmoid") {
        deriv_[k][i].SigmDer(activ_[k][i]);
      } else if (function_ == "relu") {        
        deriv_[k][i].CondAssign(activ_[k][i], 0, false, 0);
      } else {
        mexAssert(false, "LayerConv::Backward");
      }
      biases_.der(i) += deriv_[k][i].Sum();  
      for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {        
        if (prev_layer->type_ != "i" && prev_layer->type_ != "j") {
          Mat der_mat(prev_layer->mapsize_);
          Filter(deriv_[k][i], kernels_[i][j].get(), padding_der, der_mat);        
          prev_layer->deriv_[k][j] += der_mat;
        }
        Mat ker_mat(kernelsize_);
        Filter(prev_layer->activ_[k][j], deriv_[k][i], padding_, ker_mat);
        kernels_[i][j].der() += ker_mat;                  
      }      
    }        
  }
  biases_.der() /= batchsize_;
  for (size_t i = 0; i < outputmaps_; ++i) {
    for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {        
      kernels_[i][j].der() /= batchsize_;   
    }
  }
  /*
  for (int i = 0; i < 10; ++i) {
    mexPrintMsg("Conv: deriv_kern", kernels_[0][0].der()(0, i)); 
  } */
}

void LayerConv::UpdateWeights(const Params &params, bool isafter) {
  mexAssert(kernels_.size() > 0, "In LayerConv::UpdateWeights the kernels are empty");
  size_t prev_outputmaps = kernels_[0].size();
  for (size_t i = 0; i < outputmaps_; ++i) {    
    for (size_t j = 0; j < prev_outputmaps; ++j) {
      kernels_[i][j].Update(params, isafter);      
    }
  }
  biases_.Update(params, isafter);  
}

void LayerConv::GetWeights(ftype *&weights, ftype *weights_end) const {  
  //mexAssert(kernels_.size() > 0, "In LayerConv::GetWeights the kernels are empty");
  size_t numel = 1;
  for (size_t i = 0; i < numdim_; ++i) {
    numel *= kernelsize_[i];
  }  
  for (size_t i = 0; i < outputmaps_; ++i) {    
    for (size_t j = 0; j < length_prev_; ++j) {
      mexAssert(weights_end - weights >= numel,
        "In 'LayerConv::GetWeights' the vector of weights is too short!");
      kernels_[i][j].Write(weights);
      weights += numel;      
    }    
  }
  mexAssert(weights_end - weights >= outputmaps_,
      "In 'LayerConv::GetWeights' the vector of weights is too short!");
  biases_.Write(weights);      
  weights += outputmaps_;  
}

void LayerConv::SetWeights(ftype *&weights, ftype *weights_end) {

  kernels_.resize(outputmaps_);
  for (size_t i = 0; i < outputmaps_; ++i) {    
    kernels_[i].resize(length_prev_);
  }  
  size_t numel = 1;
  for (size_t i = 0; i < numdim_; ++i) {
    numel *= kernelsize_[i];
  }
  size_t fan_in = length_prev_ * numel;
  size_t fan_out = outputmaps_ * numel;
  ftype rand_coef = 2 * sqrt((ftype) 6 / (fan_in + fan_out));
  
  //mexPrintMsg("kernelsize_[0]", kernelsize_[0]);
  //mexPrintMsg("kernelsize_[1]", kernelsize_[1]);
  //mexPrintMsg("numel", numel);
  //mexPrintMsg("outputmaps_", outputmaps_);
  //mexPrintMsg("length_prev_", length_prev_);
  for (size_t i = 0; i < outputmaps_; ++i) {    
    for (size_t j = 0; j < length_prev_; ++j) {
      if (weights == NULL) {
        kernels_[i][j].Init(rand_coef, kernelsize_);      
      } else {        
        mexAssert(weights_end - weights >= numel,
          "In 'LayerConv::SetWeights' the vector of weights is too short!");
        kernels_[i][j].Init(weights, kernelsize_);
        weights += numel;
      }
    }
  }
  std::vector<size_t> biassize(2);
  biassize[0] = outputmaps_; biassize[1] = 1;  
  if (weights == NULL) {
    biases_.Init((ftype) 0, biassize);
  } else {
    mexAssert(weights_end - weights >= outputmaps_,
      "In 'LayerConv::SetWeights' the vector of weights is too short!");
    biases_.Init(weights, biassize);
    weights += outputmaps_;
  }
}  

size_t LayerConv::NumWeights() const {
  size_t num_weights = length_prev_;
  for (size_t i = 0; i < numdim_; ++i) {
    num_weights *= kernelsize_[i];
  }
  return (num_weights + 1) * outputmaps_; // +1 for biases
}
