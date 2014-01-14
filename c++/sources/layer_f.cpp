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
  function_ = "sigmoid";
  outputmaps_ = 0;
  droprate_ = 0;
  numdim_ = 1;
  batchsize_ = 0;
}  
  
void LayerFull::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(mexIsField(mx_layer, "length"), "The 'f' type layer must contain the 'length' field");
  ftype length = mexGetScalar(mexGetField(mx_layer, "length"));
  mexAssert(1 <= length, "Length on the 'f' layer must be greater or equal to 1");
  length_ = (size_t) length;  
  mapsize_.assign(prev_layer->numdim_, 0);  
  length_prev_ = prev_layer->length_;
  if (mexIsField(mx_layer, "function")) {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
  }
  if (function_ == "SVM") {
    mexAssert(mexIsField(mx_layer, "C"), "The 'SVM' layer must contain the 'C' field");
    c_ = mexGetScalar(mexGetField(mx_layer, "C"));
    mexAssert(c_ > 0, "C on the 'f' layer must be positive");
  } else {    
    mexAssert(function_ == "sigmoid" || function_ == "relu", "Unknown function for the 'f' layer");
  }
  if (mexIsField(mx_layer, "droprate")) {
    droprate_ = mexGetScalar(mexGetField(mx_layer, "droprate"));
  }
  mexAssert(0 <= droprate_ && droprate_ < 1, "Droprate must be in the range [0, 1)");  
}

void LayerFull::Forward(Layer *prev_layer, bool istrain) {
  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  if (droprate_ > 0) { // dropout
    if (istrain) { // training      
      Mat dropmat;
      dropmat.rand(batchsize_, length_prev_);
      prev_layer->activ_mat_.CondAssign(dropmat, droprate_, false, 0);
    } else { // testing
      prev_layer->activ_mat_ *= (1 - droprate_);      
    }
  }
  Prod(prev_layer->activ_mat_, false, weights_.get(), false, activ_mat_);
  activ_mat_.AddVect(biases_.get(), 2);
  if (function_ == "sigmoid") {
    activ_mat_.Sigmoid();
  } else if (function_ == "relu") {
    activ_mat_.ElemMax(0);
  } else if (function_ == "SVM") {
  } else {
    mexAssert(false, "LayerFull::Forward");
  }
  if (!istrain) prev_layer->activ_mat_.clear();
  /*
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Full: activ_[0]", activ_mat_(0, i)); 
  } */
}

void LayerFull::Backward(Layer *prev_layer) {
  if (function_ == "sigmoid") {
    deriv_mat_.SigmDer(activ_mat_);    
  } else if (function_ == "relu") {    
    deriv_mat_.CondAssign(activ_mat_, 0, false, 0);    
  } else if (function_ == "SVM") {    
  } else {
    mexAssert(false, "LayerFull::Forward");
  }
  Prod(prev_layer->activ_mat_, true, deriv_mat_, false, weights_.der());
  weights_.der() /= batchsize_;
  if (function_ == "SVM") {
    Mat weights_reg = weights_.get();    
    weights_.der() += (weights_reg /= c_);
  }  
  biases_.der() = Mean(deriv_mat_, 1);
  if (prev_layer->type_ != "i" && prev_layer->type_ != "j") {
    Prod(deriv_mat_, false, weights_.get(), true, prev_layer->deriv_mat_);
    if (prev_layer->type_ != "f") {
      prev_layer->deriv_.assign(prev_layer->batchsize_, std::vector<Mat>(prev_layer->outputmaps_));
      InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_);
    }    
  }
  /*
  for (int i = 0; i < 10; ++i) {
    mexPrintMsg("Full: deriv_weig", weights_.der()(0, i)); 
  } */
}

void LayerFull::UpdateWeights(const Params &params, bool isafter) {
  weights_.Update(params, isafter);
  biases_.Update(params, isafter);  
}

void LayerFull::GetWeights(ftype *&weights, ftype *weights_end) const { 
  
  size_t numel = length_prev_ * length_;
  mexAssert(weights_end - weights >= numel,
    "In 'LayerFull::GetWeights the vector of weights is too short!");
  weights_.Write(weights);
  weights += numel;

  mexAssert(weights_end - weights >= length_,
    "In 'LayerFull::GetWeights the vector of weights is too short!");
  biases_.Write(weights);
  weights += length_;  
}

void LayerFull::SetWeights(ftype *&weights, ftype *weights_end) {
  
  std::vector<size_t> weightssize(2);
  weightssize[0] = length_prev_; weightssize[1] = length_;
  if (weights == NULL) {
    ftype rand_coef = 2 * sqrt((ftype) 6 / (length_prev_ + length_));  
    weights_.Init(rand_coef, weightssize);
  } else {
    size_t numel = length_prev_ * length_;
    mexAssert(weights_end - weights >= numel,
      "In 'LayerFull::SetWeights the vector of weights is too short!");  
    weights_.Init(weights, weightssize);      
    weights += numel;
  }
  
  std::vector<size_t> biassize(2);
  biassize[0] = 1; biassize[1] = length_;    
  if (weights == NULL) {
    biases_.Init((ftype) 0, biassize);
  } else {
    mexAssert(weights_end - weights >= length_,
      "In 'LayerFull::SetWeights the vector of weights is too short!");  
    biases_.Init(weights, biassize);
    weights += length_;
  }

  //mexPrintMsg("length_prev_", length_prev_);
  //mexPrintMsg("length_", length_);
  
}

size_t LayerFull::NumWeights() const {
  return (length_prev_ + 1) * length_; // +1 for biases
}

