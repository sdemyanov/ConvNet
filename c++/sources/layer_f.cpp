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

#include "layer_f.h"

LayerFull::LayerFull() {
  type_ = "f";    
  function_ = "relu";
  outputmaps_ = 0;
  dropout_ = 0;
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
    if (function_ == "SVM") {
      mexAssert(mexIsField(mx_layer, "C"), "The 'SVM' layer must contain the 'C' field");
      c_ = mexGetScalar(mexGetField(mx_layer, "C"));
      mexAssert(c_ > 0, "C on the 'f' layer must be positive");
    }
    mexAssert(function_ == "soft" || function_ == "sigm" || 
              function_ == "relu" || function_ == "SVM",
      "Unknown function for the 'f' layer");
  }
  if (mexIsField(mx_layer, "dropout")) {
    dropout_ = mexGetScalar(mexGetField(mx_layer, "dropout"));
  }
  mexAssert(0 <= dropout_ && dropout_ < 1, "Dropout must be in the range [0, 1)");  
}

void LayerFull::Forward(Layer *prev_layer, int passnum) {
  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  if (dropout_ > 0) { // dropout
    if (passnum == 1) { // training      
      Mat dropmat(batchsize_, length_prev_);
      dropmat.rand();
      prev_layer->activ_mat_.CondAssign(dropmat, dropout_, false, 0);
    } else if (passnum == 0) { // testing
      prev_layer->activ_mat_ *= (1 - dropout_);      
    }
  }  
  Prod(prev_layer->activ_mat_, false, weights_.get(), true, activ_mat_);
  if (passnum == 0 || passnum == 1) {
    activ_mat_.AddVect(biases_.get(), 2);    
  }
  activ_mat_.Validate();
  /*
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Full: activ_[0]", activ_mat_(0, i)); 
  } */
}

void LayerFull::Backward(Layer *prev_layer) {  
  Prod(deriv_mat_, false, weights_.get(), false, prev_layer->deriv_mat_);
  if (prev_layer->type_ != "f") {
    InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_);
  }
  prev_layer->deriv_mat_.Validate();  
}

void LayerFull::CalcWeights(Layer *prev_layer) {
  biases_.der() = Mean(deriv_mat_, 1);  
  Prod(deriv_mat_, true, prev_layer->activ_mat_, false, weights_.der());
  weights_.der() /= batchsize_;
  if (function_ == "SVM") {
    Mat weights_reg = weights_.get();    
    weights_.der() += (weights_reg /= c_);
  }  
  biases_.der().Validate();
  weights_.der().Validate();
  /*
  for (int i = 0; i < 10; ++i) {
    mexPrintMsg("Full: deriv_weig", weights_.der()(0, i)); 
  } */
}

void LayerFull::CalcWeights2(Layer *prev_layer, const std::vector<size_t> &invalid) {
  if (invalid.size() == 0) {
    Prod(activ_mat_, true, prev_layer->deriv_mat_, false, weights_.der2());
    weights_.der2() /= batchsize_;  
  } else {
    if (invalid.size() == batchsize_) {
      weights_.der2().assign(0);
      return;
    }
    std::vector<size_t> valid(batchsize_ - invalid.size());
    size_t invind = 0;
    for (size_t i = 0; i < batchsize_; ++i) {
      if (i == invalid[invind]) {
        invind++;
      } else {
        valid[i - invind] = i;
      }
    }
    batchsize_ -= invalid.size();
    Mat activ_mat = SubMat(activ_mat_, valid, 1);
    Mat deriv_mat_prev = SubMat(prev_layer->deriv_mat_, valid, 1);    
    Prod(activ_mat, true, deriv_mat_prev, false, weights_.der2());
    weights_.der2() /= batchsize_;  
  }
  weights_.der2().Validate();
}

void LayerFull::UpdateWeights(const Params &params, size_t epoch, bool isafter) {
  weights_.Update(params, epoch, isafter);
  biases_.Update(params, epoch, isafter);  
}

void LayerFull::SetWeights(ftype *&weights, bool isgen) {
  
  std::vector<size_t> weightssize(2);
  weightssize[0] = length_; weightssize[1] = length_prev_;
  if (isgen) {
    ftype rand_coef = 2 * sqrt((ftype) 6 / (length_prev_ + length_));  
    weights_.Init(weights, weightssize, rand_coef);
  } else {
    size_t numel = length_ * length_prev_;
    weights_.Init(weights, weightssize);      
    weights += numel;
  }
  
  std::vector<size_t> biassize(2);
  biassize[0] = 1; biassize[1] = length_;    
  if (isgen) {
    biases_.Init(weights, biassize, 0);
  } else {
    biases_.Init(weights, biassize);
    weights += length_;
  }

  //mexPrintMsg("length_prev_", length_prev_);
  //mexPrintMsg("length_", length_);
  
}

size_t LayerFull::NumWeights() const {
  return (length_prev_ + 1) * length_; // +1 for biases
}

