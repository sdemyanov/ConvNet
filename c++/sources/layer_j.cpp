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

#include "layer_j.h"

LayerJitt::LayerJitt() {
  type_ = "j";
  is_weights_ = false;
  batchsize_ = 0;
}  
  
void LayerJitt::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  //mexAssert(prev_layer->type_ == "i", "The 'j' type layer must be after the input one");
  numdim_ = prev_layer->numdim_;
  outputmaps_ = prev_layer->outputmaps_;
  length_prev_ = prev_layer->length_prev_;
  mexAssert(mexIsField(mx_layer, "mapsize"), "The 'j' type layer must contain the 'mapsize' field");
  std::vector<ftype> mapsize = mexGetVector(mexGetField(mx_layer, "mapsize"));  
  mexAssert(mapsize.size() == numdim_, "Length of jitter mapsize vector and maps dimensionality must coincide");
  mapsize_.resize(numdim_);
  length_ = outputmaps_;
  for (size_t i = 0; i < numdim_; ++i) {
    mexAssert(mapsize[i] <= prev_layer->mapsize_[i], "In 'j' layer new mapsize cannot be larger than the old one");    
    mapsize_[i] = (size_t) mapsize[i];
    length_ *= mapsize_[i];
  }
  shift_.assign(numdim_, 0);
  if (mexIsField(mx_layer, "shift")) {
    shift_ = mexGetVector(mexGetField(mx_layer, "shift"));  
    mexAssert(shift_.size() == numdim_, "Length of jitter shift vector and maps dimensionality must coincide");
    for (size_t i = 0; i < numdim_; ++i) {      
      mexAssert(0 <= shift_[i] && shift_[i] < mapsize_[i], "Shift in 'j' layer is out of range");
    }
  }
  scale_.assign(numdim_, 1);
  if (mexIsField(mx_layer, "scale")) {
    scale_ = mexGetVector(mexGetField(mx_layer, "scale"));  
    mexAssert(scale_.size() == numdim_, "Length of jitter scale vector and maps dimensionality must coincide");
    for (size_t i = 0; i < numdim_; ++i) {      
      mexAssert(1 <= scale_[i] && scale_[i] < mapsize_[i], "Scale in 'j' layer is out of range");
    }
  }
  mirror_.assign(numdim_, false);
  if (mexIsField(mx_layer, "mirror")) {
    std::vector<ftype> mirror = mexGetVector(mexGetField(mx_layer, "mirror"));  
    mexAssert(mirror.size() == numdim_, "Length of jitter scale vector and maps dimensionality must coincide");
    for (size_t i = 0; i < numdim_; ++i) {      
      mirror_[i] = (mirror[i] > 0);      
    }
  }
  angle_ = 0;
  if (mexIsField(mx_layer, "angle")) {
    angle_ = mexGetScalar(mexGetField(mx_layer, "angle"));    
    mexAssert(0 <= angle_ && angle_ <= 1, "Angle in 'j' layer must be between 0 and 1");    
  }  
  default_ = 0;
  if (mexIsField(mx_layer, "default")) {
    default_ = mexGetScalar(mexGetField(mx_layer, "default"));    
  } else {  
    // check that the transformed image is always inside the original one
    std::vector<ftype> maxsize(numdim_, 0);    
    for (size_t i = 0; i < numdim_; ++i) {
      maxsize[i] = (ftype) (mapsize_[i] - 1) * scale_[i];      
    }
    if (angle_ > 0) {
      ftype angle_inn = atan2((ftype) mapsize_[0], (ftype) mapsize_[1]) / M_PI;    
      ftype maxsin = 1;
      if (angle_inn + angle_ < 0.5) {
        maxsin = sin(M_PI * (angle_inn + angle_));        
      }    
      ftype maxcos = 1;
      if (angle_inn > angle_) {
        maxcos = cos(M_PI * (angle_inn - angle_));
      }    
      ftype maxrad = sqrt(maxsize[0]*maxsize[0] + maxsize[1]*maxsize[1]);  
      maxsize[0] = maxrad * maxsin;
      maxsize[1] = maxrad * maxcos;    
    }
    std::vector<ftype> oldmapsize(numdim_, 0);
    for (size_t i = 0; i < numdim_; ++i) { 
      oldmapsize[i] = (ftype) prev_layer->mapsize_[i];
    }
    ftype min0 = (oldmapsize[0]/2 - 0.5) - maxsize[0]/2 - shift_[0];
    ftype max0 = (oldmapsize[0]/2 - 0.5) + maxsize[0]/2 + shift_[0];
    ftype min1 = (oldmapsize[1]/2 - 0.5) - maxsize[1]/2 - shift_[1];
    ftype max1 = (oldmapsize[1]/2 - 0.5) + maxsize[1]/2 + shift_[1];
    if (!(0 <= min0 && max0 < oldmapsize[0] && 0 <= min1 && max1 < oldmapsize[1])) {
      mexPrintMsg("min1", min0); mexPrintMsg("max1", max0);
      mexPrintMsg("min2", min1); mexPrintMsg("max2", max1);    
      mexAssert(false, "For these jitter parameters the new image is out of the original image");
    }
  }
} 

void LayerJitt::Forward(Layer *prev_layer, int passnum) {
  batchsize_ = prev_layer->batchsize_;
  activ_mat_.resize(batchsize_, length_);
  InitMaps(activ_mat_, mapsize_, activ_);  
  #if USE_MULTITHREAD == 1
    #pragma omp parallel for
  #endif
  for (int k = 0; k < batchsize_; ++k) {  
    for (size_t i = 0; i < outputmaps_; ++i) {
      std::vector<ftype> shift(numdim_, 0);
      std::vector<ftype> scale(numdim_, 1);
      std::vector<bool> mirror(numdim_, false);
      ftype angle = 0;
      for (size_t j = 0; j < numdim_; ++j) {
        if (shift_[j] > 0) {
          shift[j] = ((ftype) rand() / RAND_MAX * 2 - 1) * shift_[j];          
        }
        if (scale_[j] > 1) {
          scale[j] = pow(scale_[j], (ftype) rand() / RAND_MAX * 2 - 1);          
        }
        if (mirror_[j]) {
          mirror[j] = ((ftype) rand() / RAND_MAX > 0.5);          
        }        
      }
      if (angle_ > 0) {
        angle = ((ftype) rand() / RAND_MAX * 2 - 1) * M_PI * angle_;
      }      
      Transform(prev_layer->activ_[k][i], shift, scale, mirror, angle, default_, activ_[k][i]);      
    }    
  }
  activ_mat_.Validate();  
  /*
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Jitt: activ_[0][0]", activ_[0][0](0, i)); 
  } */
}
