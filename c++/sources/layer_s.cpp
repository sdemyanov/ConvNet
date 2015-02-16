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

#include "layer_s.h"

LayerScal::LayerScal() {
  type_ = "s";
  function_ = "mean";   
  batchsize_ = 0;  
}  
  
void LayerScal::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer->type_ != "f", "The 's' type layer cannot be after 'f' type layer");
  numdim_ = prev_layer->numdim_;
  outputmaps_ = prev_layer->outputmaps_;
  length_prev_ = prev_layer->length_prev_; // not used
  mexAssert(mexIsField(mx_layer, "scale"), "The 's' type layer must contain the 'scale' field");
  std::vector<ftype> scale = mexGetVector(mexGetField(mx_layer, "scale"));  
  mexAssert(scale.size() == numdim_, "Length of the scale vector and maps dimensionality must coincide");
  scale_.resize(numdim_);
  stride_.resize(numdim_);
  mapsize_.resize(numdim_);
  length_ = outputmaps_;
  for (size_t i = 0; i < numdim_; ++i) {
    mexAssert(1 <= scale[i] && scale[i] <= prev_layer->mapsize_[i], "Scale on the 's' layer must be in the range [1, previous_layer_mapsize]");
    scale_[i] = (size_t) scale[i];
    stride_[i] = scale_[i];
    mapsize_[i] = DIVUP(prev_layer->mapsize_[i], stride_[i]);    
    length_ *= mapsize_[i];
  }  
  #if COMP_REGIME == 2	// GPU
    mexAssert(scale_[0] == scale_[1], "In the GPU version the 'scale' should be squared on all layers");
  #endif
  if (mexIsField(mx_layer, "stride")) {
    std::vector<ftype> stride = mexGetVector(mexGetField(mx_layer, "stride"));
    mexAssert(stride.size() == numdim_, "Stride vector has the wrong length");
    length_ = outputmaps_;
    for (size_t i = 0; i < numdim_; ++i) {
      mexAssert(1 <= stride[i] && stride[i] <= prev_layer->mapsize_[i], "Stride on the 's' layer must be in the range [1, previous_layer_mapsize]");    
      stride_[i] = (size_t) stride[i];      
      #if COMP_REGIME == 2	// GPU
        mexAssert(stride_[i] <= scale_[i], "In the GPU version the 'stride' should less or equal than 'scale'");      
      #endif
      mapsize_[i] = DIVUP(prev_layer->mapsize_[i], stride_[i]);
      length_ *= mapsize_[i];
    }
    #if COMP_REGIME == 2	// GPU
      mexAssert(stride_[0] == stride_[1], "In the GPU version the 'stride' should be squared on all layers");      
    #endif
  }  
  if (mexIsField(mx_layer, "function")) {
    function_ = mexGetString(mexGetField(mx_layer, "function"));
    mexAssert(function_ == "max" || function_ == "mean", "Unknown function for the 's' layer");    
  }
}
    
void LayerScal::Forward(Layer *prev_layer, int passnum) {  
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  #if COMP_REGIME != 2
    std::vector< std::vector<Mat> > prev_activ, activ, prev_deriv, deriv;    
    InitMaps(prev_layer->activ_mat_, prev_layer->mapsize_, prev_activ);
    InitMaps(activ_mat_, mapsize_, activ);    
    if (function_ == "max" && passnum == 3) {
      InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_deriv);
      InitMaps(deriv_mat_, mapsize_, deriv);    
    }
    #if COMP_REGIME == 1
      #pragma omp parallel for
    #endif  
    for (int k = 0; k < batchsize_; ++k) {      
      for (size_t i = 0; i < outputmaps_; ++i) {
        if (function_ == "mean") {
          MeanScale(prev_activ[k][i], scale_, stride_, activ[k][i]);
        } else if (function_ == "max") {          
          if (passnum == 0 || passnum == 1) {
            MaxScale(prev_activ[k][i], scale_, stride_, activ[k][i]);       
          } else if (passnum == 3) {
            MaxScaleDer(prev_deriv[k][i], deriv[k][i],
                        activ[k][i], prev_activ[k][i],
                        scale_, stride_, false);
          }
        }
      }    
    }    
  #else // GPU  
    if (function_ == "mean") {
      AvgPooling(prev_layer->activ_mat_, activ_mat_,
                 prev_layer->mapsize_, scale_[0], stride_[0]);            
    } else if (function_ == "max") {
      if (passnum == 0 || passnum == 1) {
        MaxPooling(prev_layer->activ_mat_, activ_mat_,
                   prev_layer->mapsize_, scale_[0], stride_[0]);        
      } else if (passnum == 3) {  
        MaxPoolingUndo(prev_layer->deriv_mat_, deriv_mat_,
                       activ_mat_, prev_layer->activ_mat_,
                       prev_layer->mapsize_, scale_[0], stride_[0], false);          
      }
    }
  #endif
}

void LayerScal::Backward(Layer *prev_layer) {    
  prev_layer->deriv_mat_.resize(prev_layer->batchsize_, prev_layer->length_);
  #if COMP_REGIME != 2
    std::vector< std::vector<Mat> > prev_activ, activ, prev_deriv, deriv;    
    InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_deriv);
    InitMaps(deriv_mat_, mapsize_, deriv);    
    if (function_ == "max") {
      InitMaps(prev_layer->activ_mat_, prev_layer->mapsize_, prev_activ);
      InitMaps(activ_mat_, mapsize_, activ);    
    }
    #if COMP_REGIME == 1
      #pragma omp parallel for
    #endif
    for (int k = 0; k < batchsize_; ++k) {
      for (size_t i = 0; i < outputmaps_; ++i) {
        if (function_ == "mean") {
          MeanScaleDer(deriv[k][i], scale_, stride_, prev_deriv[k][i]);
        } else if (function_ == "max") {
          MaxScaleDer(prev_activ[k][i], activ[k][i], 
                      deriv[k][i], prev_deriv[k][i],
                      scale_, stride_, true);
        }      
      }
    }  
  #else // GPU    
    if (function_ == "mean") {
      AvgPoolingUndo(deriv_mat_, prev_layer->deriv_mat_, 
                     prev_layer->mapsize_, scale_[0], stride_[0]);      
    } else if (function_ == "max") {
      MaxPoolingUndo(prev_layer->activ_mat_, activ_mat_,
                     deriv_mat_, prev_layer->deriv_mat_,
                     prev_layer->mapsize_, scale_[0], stride_[0], true);  
    }
  #endif  
  //InitMaps(prev_layer->deriv_mat_, prev_layer->mapsize_, prev_layer->deriv_);  
  //InitMaps(deriv_mat_, mapsize_, deriv_);  
  //mexPrintMsg("Sum: deriv_[b][o]", deriv_[batchsize_ - 1][outputmaps_ - 1].sum()); 
  //mexPrintMsg("Sum: prev_layer->deriv_[b][o]", prev_layer->deriv_[batchsize_ - 1][outputmaps_ - 1].sum()); 
  /*
  if (print == 1) {
  mexPrintMsg("SCALE BACKWARD");    
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
