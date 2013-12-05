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

#include "layer_f.h"

LayerFull::LayerFull() {
  type_ = "f";    
}  
  
void LayerFull::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(mexIsField(mx_layer, "length"), "The 'f' type layer must contain the 'length' field");
  length_ = (size_t) mexGetScalar(mexGetField(mx_layer, "length"));
  mexAssert(1 <= length_, "Length on the 'f' layer must be greater or equal to 1");
  if (prev_layer->type_ != "f") {
    length_prev_ = prev_layer->mapsize_[0] * prev_layer->mapsize_[1] * prev_layer->outputmaps_;
  } else {
    length_prev_ = ((LayerFull*) prev_layer)->length_;
  }        
  outputmaps_ = 0;
  mapsize_.resize(prev_layer->mapsize_.size());
  mapsize_[0] = mapsize_[1] = 0;
  if (!mexIsField(mx_layer, "function")) {
    function_ = "sigmoid";
  } else {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
  }
  if (function_ == "SVM") {
    mexAssert(mexIsField(mx_layer, "C"), "The 'SVM' layer must contain the 'C' field");
    c_ = mexGetScalar(mexGetField(mx_layer, "C"));
    mexAssert(c_ > 0, "C on the 'f' layer must be positive");
  } else {
    std::string errmsg = function_ + " - unknown function for the layer";      
    mexAssert(function_ == "sigmoid" || function_ == "relu", errmsg);
  }
  if (!mexIsField(mx_layer, "droprate")) {
    droprate_ = 0;
  } else {
    droprate_ = mexGetScalar(mexGetField(mx_layer, "droprate"));
  }
  mexAssert(0 <= droprate_, "Droprate must be non-negative");
  mexAssert(droprate_ < 1 - (float) 1/length_, "Droprate must be smaller than 1-(1/length)");
  double rand_coef = 2 * sqrt((double) 6 / (length_ + length_prev_));  
  std::vector<size_t> lengthsize(2);
  lengthsize[0] = length_; lengthsize[1] = length_prev_;
  weights_.Init(lengthsize, rand_coef);
  std::vector<size_t> biassize(2);
  biassize[0] = length_; biassize[1] = 1;
  biases_.Init(biassize, 0);
}

void LayerFull::Forward(const Layer *prev_layer, bool istrain) {
  
  // concatenate all end layer feature maps into vector for each object
  batchsize_ = prev_layer->batchsize_;
  if (prev_layer->type_ != "f") {    
    input_.ReshapeFrom(prev_layer->activ_);
  } else {    
    input_ = static_cast<const LayerFull*>(prev_layer)->output_;    
  }  
  // processing the layer
  if (istrain) { // dropout, training
    std::vector<size_t> dropsize(2);
    dropsize[0] = length_prev_; dropsize[1] = batchsize_;
    Mat dropmat(dropsize);
    dropmat.Rand();
    input_.CondAssign(dropmat, droprate_, false, 0);    
  } else { // dropout, testing
    input_ *= (1 - droprate_);
  }  
  
  Prod(weights_.get(), input_, output_);
  output_.AddVect(biases_.get(), 1);
  if (function_ == "sigmoid") {
    output_.Sigmoid();
  } else if (function_ == "relu") {
    output_.ElemMax(0);
  } else if (function_ == "SVM") {
  } else {
    mexAssert(false, "LayerFull::Forward");
  }  
}

void LayerFull::Backward(Layer *prev_layer) {
  
  if (function_ == "sigmoid") {
    output_der_.SigmDer(output_);    
  } else if (function_ == "relu") {    
    output_der_.CondAssign(output_, 0, false, 0);    
  } else if (function_ == "SVM") {    
  } else {
    mexAssert(false, "LayerFull::Forward");
  }
  input_.Trans();
  Prod(output_der_, input_, weights_.der());
  weights_.der() /= batchsize_;
  if (function_ == "SVM") {
    Mat weights_reg = weights_.get();    
    weights_.der() += (weights_reg /= c_);
  }
  output_der_.Mean(2, biases_.der());
  Mat weights_tr;
  Trans(weights_.get(), weights_tr);
  Prod(weights_tr, output_der_, input_der_);
  if (prev_layer->type_ != "f") {    
    input_der_.ReshapeTo(prev_layer->deriv_, prev_layer->outputmaps_, 
                         prev_layer->batchsize_, prev_layer->mapsize_);
  } else {
    static_cast<LayerFull*>(prev_layer)->output_der_ = input_der_;    
  }  
}

void LayerFull::UpdateWeights(const Params &params, bool isafter) {
  weights_.Update(params, isafter);
  biases_.Update(params, isafter);  
}

void LayerFull::GetWeights(std::vector<double> &weights) const {  
  std::vector<double> curweights = weights_.get().ToVect();
  weights.insert(weights.end(), curweights.begin(), curweights.end());
  std::vector<double> curbiases = biases_.get().ToVect();
  weights.insert(weights.end(), curbiases.begin(), curbiases.end());  
}

void LayerFull::SetWeights(std::vector<double> &weights) {
  
  size_t numel = length_ * length_prev_;
  mexAssert(weights.size() >= numel, "Vector of weights is too short!");
  std::vector<double> curweights(weights.begin(), weights.begin() + numel);  
  weights_.get().FromVect(curweights, weights_.size());      
  weights.erase(weights.begin(), weights.begin() + numel);
  
  mexAssert(weights.size() >= length_, "Vector of weights is too short!");
  std::vector<double> curbiases(weights.begin(), weights.begin() + length_);  
  biases_.get().FromVect(curbiases, biases_.size());
  weights.erase(weights.begin(), weights.begin() + length_);        
}

