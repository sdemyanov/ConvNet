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

LayerConv::LayerConv() {
  type_ = "c";
}
  
void LayerConv::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 'c' type layer cannot be after 'f' type layer");
  mexAssert(mexIsField(mx_layer, "kernelsize"), "The 'c' type layer must contain the 'kernelsize' field");
  std::vector<double> kernelsize = mexGetVector(mexGetField(mx_layer, "kernelsize"));
  kernelsize_.resize(kernelsize.size());
  mapsize_.resize(prev_layer->mapsize_.size());
  mexAssert(mapsize_.size() == kernelsize_.size(), "Kernels and maps must be the same dimensionality");
  for (size_t i = 0; i < mapsize_.size(); ++i) {
    kernelsize_[i] = (size_t) kernelsize[i];
    mexAssert(1 <= kernelsize_[i], "Kernelsize on the 'c' layer must be greater or equal to 1");
    mapsize_[i] = prev_layer->mapsize_[i] - kernelsize_[i] + 1;
    mexAssert(1 <= mapsize_[i], "Kernelsize on the 'c' layer must be smaller or equal to mapsize");
  }
  mexAssert(mexIsField(mx_layer, "outputmaps"), "The 'c' type layer must contain the 'outputmaps' field");
  outputmaps_ = (size_t) mexGetScalar(mexGetField(mx_layer, "outputmaps"));  
  mexAssert(1 <= outputmaps_, "Outputmaps on the 'i' layer must be greater or equal to 1");
  if (!mexIsField(mx_layer, "function")) {
    function_ = "sigmoid";
  } else {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
  }
  std::string errmsg = function_ + " - unknown function for the layer";      
  mexAssert(function_ == "sigmoid" || function_ == "relu", errmsg);    
  
  activ_.resize(outputmaps_);
  deriv_.resize(outputmaps_);
  std::vector<size_t> biassize(2);
  biassize[0] = outputmaps_; biassize[1] = 1;
  biases_.Init(biassize, 0);
  kernels_.resize(outputmaps_);
  size_t fan_in = prev_layer->outputmaps_ * kernelsize_[0] * kernelsize_[1];
  size_t fan_out = outputmaps_ * kernelsize_[0] * kernelsize_[1];
  double rand_coef = 2 * sqrt((double) 6 / (fan_in + fan_out));
  for (size_t i = 0; i < outputmaps_; ++i) {    
    kernels_[i].resize(prev_layer->outputmaps_);
    for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {
      kernels_[i][j].Init(kernelsize_, rand_coef);      
    }
  }  
}
    
void LayerConv::Forward(const Layer *prev_layer, bool istrain) {
  
  batchsize_ = prev_layer->batchsize_;
  for (size_t i = 0; i < outputmaps_; ++i) {
    activ_[i].resize(batchsize_);
    for (size_t k = 0; k < batchsize_; ++k) {
      activ_[i][k].init(mapsize_, biases_.get(i));
      for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {
        Mat conv_act(mapsize_);
        prev_layer->activ_[j][k].Filter(kernels_[i][j].get(), conv_act, false);
        activ_[i][k] += conv_act;
      }
      if (function_ == "sigmoid") {
        activ_[i][k].Sigmoid();
      } else if (function_ == "relu") {
        activ_[i][k].ElemMax(0);
      } else {
        mexAssert(false, "LayerConv::Forward");
      }      
    } // objects        
  } // outputmaps  
}

void LayerConv::Backward(Layer *prev_layer) {  
  
  for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {    
    prev_layer->deriv_[j].resize(batchsize_);
    for (size_t k = 0; k < batchsize_; ++k) {
      prev_layer->deriv_[j][k].init(prev_layer->mapsize_, 0);      
    }
  }  
  for (size_t i = 0; i < outputmaps_; ++i) {    
    biases_.der(i) = 0;
    for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {
      kernels_[i][j].der().assign(0);      
    }    
    for (size_t k = 0; k < batchsize_; ++k) {      
      if (function_ == "sigmoid") {
        deriv_[i][k].SigmDer(activ_[i][k]);
      } else if (function_ == "relu") {        
        deriv_[i][k].CondAssign(activ_[i][k], 0, false, 0);
      } else {
        mexAssert(false, "LayerConv::Backward");
      }      
      biases_.der(i) += deriv_[i][k].Sum();  
      for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {        
        Mat conv_der(prev_layer->mapsize_);
        deriv_[i][k].Filter(kernels_[i][j].get(), conv_der, true);        
        prev_layer->deriv_[j][k] += conv_der;                
        Mat conv_ker(kernelsize_);
        prev_layer->activ_[j][k].Filter(deriv_[i][k], conv_ker, false);
        kernels_[i][j].der() += conv_ker;                  
      }      
    }    
    biases_.der(i) /= batchsize_;
    for (size_t j = 0; j < prev_layer->outputmaps_; ++j) {
      kernels_[i][j].der() /= batchsize_;      
    }    
  }
  //mexPrintMsg("kern", kernels_[i][j].get().Sum());  
  //mexPrintMsg("kern v", kernels_[i][j].get()(0, kernels_[i][j].get().size2()-1));          
  //mexPrintMsg("prev", prev_layer->deriv_[0][0].Sum());    
  //mexPrintMsg("prev v", prev_layer->deriv_[0][0](0, prev_layer->deriv_[0][0].size2()-1));      
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

void LayerConv::GetWeights(std::vector<double> &weights) const {  
  mexAssert(kernels_.size() > 0, "In LayerConv::GetWeights the kernels are empty");
  size_t prev_outputmaps = kernels_[0].size();  
  for (size_t i = 0; i < outputmaps_; ++i) {    
    for (size_t j = 0; j < prev_outputmaps; ++j) {
      std::vector<double> curweights = kernels_[i][j].get().ToVect();      
      weights.insert(weights.end(), curweights.begin(), curweights.end());      
    }    
  }
  std::vector<double> curbiases = biases_.get().ToVect();      
  weights.insert(weights.end(), curbiases.begin(), curbiases.end());    
}

void LayerConv::SetWeights(std::vector<double> &weights) {
  mexAssert(kernels_.size() > 0, "In LayerConv::SetWeights the kernels are empty");
  size_t prev_outputmaps = kernels_[0].size();
  size_t numel = kernelsize_[0] * kernelsize_[1];
  for (size_t i = 0; i < outputmaps_; ++i) {    
    for (size_t j = 0; j < prev_outputmaps; ++j) {
      mexAssert(weights.size() >= numel, "Vector of weights is too short!");
      std::vector<double> curweights(weights.begin(), weights.begin() + numel);
      kernels_[i][j].get().FromVect(curweights, kernelsize_);
      //mexPrintMsg("i", i);  
      //mexPrintMsg("j", j);  
      //mexPrintMsg("kern", kernels_[i][j].der().Sum());  
      weights.erase(weights.begin(), weights.begin() + numel);      
    }    
  }
  mexAssert(weights.size() >= outputmaps_, "Vector of weights is too short!");
  std::vector<double> curbiases(weights.begin(), weights.begin() + outputmaps_);  
  biases_.get().FromVect(curbiases, biases_.size());      
  weights.erase(weights.begin(), weights.begin() + outputmaps_);      
}

  
