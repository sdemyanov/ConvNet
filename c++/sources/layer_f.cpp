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
      dropmat_.resize(batchsize_, length_prev_ * length_);
      dropmat_.rand();
      dropmat_.CondAssign(dropmat_, false, dropout_, 0);
      dropmat_.CondAssign(dropmat_, true, 0, 1);
      Prod(prev_layer->activ_mat_, false, weights_.get(), true, dropmat_, activ_mat_);
      
      dropmat_bias_.resize(batchsize_, length_);
      dropmat_bias_.rand();
      dropmat_bias_.CondAssign(dropmat_bias_, false, dropout_, 0);
      dropmat_bias_.CondAssign(dropmat_bias_, true, 0, 1);      
      activ_mat_.AddVect(biases_.get(), dropmat_bias_, 1);
    } else if (passnum == 0) { // testing
      prev_layer->activ_mat_ *= (1 - dropout_);
      Prod(prev_layer->activ_mat_, false, weights_.get(), true, activ_mat_);      
      Mat biases = biases_.get();
      biases *= (1 - dropout_);
      activ_mat_.AddVect(biases, 1);      
    }
  } else {
    Prod(prev_layer->activ_mat_, false, weights_.get(), true, activ_mat_);    
    if (passnum == 0 || passnum == 1) {
      activ_mat_.AddVect(biases_.get(), 1);            
    }    
  }  
  activ_mat_.Validate();
  /*
  if (print == 1) {
  mexPrintMsg("FULL");    
  Mat m;
  m.attach(activ_mat_);
  mexPrintMsg("s1", m.size1());    
  mexPrintMsg("s2", m.size2()); 
  mexPrintMsg("totalsum", m.sum());    
  Mat versum = Sum(m, 1);
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("versum", versum(0, i));    
  }
  Mat horsum = Sum(m, 2);
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("horsum", horsum(i, 0));    
  }  
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Horizontal", m(0, i));    
  }
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Vertical", m(i, 0));    
  }
  }*/
}

void LayerFull::Backward(Layer *prev_layer) {
  prev_layer->deriv_mat_.resize(batchsize_, prev_layer->length_);
  if (dropout_ > 0) {
    Prod(deriv_mat_, false, weights_.get(), false, dropmat_, prev_layer->deriv_mat_);
  } else {
    Prod(deriv_mat_, false, weights_.get(), false, prev_layer->deriv_mat_);
  }
  prev_layer->deriv_mat_.Validate();   
  /*
  if (print == 1) {
  mexPrintMsg("FULL BACKWARD");    
  Mat m;
  m.attach(prev_layer->deriv_mat_);
  mexPrintMsg("s1", m.size1());    
  mexPrintMsg("s2", m.size2()); 
  mexPrintMsg("totalsum", m.sum());    
  Mat versum = Sum(m, 1);
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("versum", versum(0, i));    
  }
  Mat horsum = Sum(m, 2);
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("horsum", horsum(i, 0));    
  }  
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Horizontal", m(0, i));    
  }
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Vertical", m(i, 0));    
  }
  }*/
}

void LayerFull::CalcWeights(Layer *prev_layer, int passnum) {

  if (passnum < 2) return;
  Mat weights_der;
  if (passnum == 2) {
    weights_der.attach(weights_.der());
  }
  if (dropout_ > 0) {    
    Prod(deriv_mat_, true, prev_layer->activ_mat_, false, dropmat_, weights_der);    
    Mat dividers(1, dropmat_.size2());
    Sum(dropmat_, dividers, 1);
    dividers.reshape(length_, length_prev_);
    dividers.CondAssign(dividers, false, 0, 1); // to avoid division by zero    
    weights_der /= dividers;    
    if (passnum == 2) {
      Sum(deriv_mat_, biases_.der(), 1);
      Mat dividers_bias(1, dropmat_bias_.size2());
      Sum(dropmat_bias_, dividers_bias, 1);
      dividers_bias.CondAssign(dividers_bias, false, 0, 1);
      biases_.der() /= dividers_bias;
    }    
  } else {
    Prod(deriv_mat_, true, prev_layer->activ_mat_, false, weights_der);
    weights_der /= batchsize_;    
    if (passnum == 2) {
      Mean(deriv_mat_, biases_.der(), 1);    
    }
  }
  if (passnum == 2) {    
    if (function_ == "SVM") {
      Mat weights_reg = weights_.get();    
      weights_der += (weights_reg /= c_);
    }  
    biases_.der().Validate();
  }
  weights_der.Validate();
}

void LayerFull::InitWeights(Weights &weights, size_t &offset, bool isgen) {
  
  weights_.Attach(weights, offset, length_, length_prev_, kMatlabOrder);  
  offset += length_ * length_prev_;
  if (isgen) {
    ftype rand_coef = 2 * sqrt((ftype) 6 / (length_prev_ + length_));  
    (weights_.get().rand() -= 0.5) *= rand_coef;    
  }
  biases_.Attach(weights, offset, 1, length_, kMatlabOrder);  
  offset += length_;
  if (isgen) {
    biases_.get().assign(0);
  }
}

void LayerFull::GetWeights(Mat &weights, size_t &offset) const {
  Mat weights_mat;
  weights_mat.attach(weights, offset, length_, length_prev_, kMatlabOrder);
  weights_mat = weights_.get();
  offset += length_ * length_prev_;
  
  Mat biases_mat;
  biases_mat.attach(weights, offset, 1, length_, kMatlabOrder);
  biases_mat = biases_.get();
  offset += length_;  
}

size_t LayerFull::NumWeights() const {
  return (length_prev_ + 1) * length_; // +1 for biases
}
