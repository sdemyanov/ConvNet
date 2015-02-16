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
  function_ = "none";  
  batchsize_ = 0;
  randtest_ = false;
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
    mexAssert(1 <= mapsize[i], "In 'j' layer mapsize must be positive");    
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
  defval_ = 0;
  if (mexIsField(mx_layer, "defval")) {
    defval_ = mexGetScalar(mexGetField(mx_layer, "defval"));    
  } else {  
    // check that the transformed image is always inside the original one
    std::vector<ftype> maxsize(numdim_, 0);    
    for (size_t i = 0; i < numdim_; ++i) {
      maxsize[i] = (ftype) (mapsize_[i] - 1) * scale_[i];      
    }
    if (angle_ > 0) {
      ftype angle_inn = atan2((ftype) mapsize_[0], (ftype) mapsize_[1]) / kPi;    
      ftype maxsin = 1;
      if (angle_inn + angle_ < 0.5) {
        maxsin = sin(kPi * (angle_inn + angle_));        
      }    
      ftype maxcos = 1;
      if (angle_inn > angle_) {
        maxcos = cos(kPi * (angle_inn - angle_));
      }    
      ftype maxrad = (ftype) sqrt((double) (maxsize[0]*maxsize[0] + maxsize[1]*maxsize[1]));  
      maxsize[0] = maxrad * maxsin;
      maxsize[1] = maxrad * maxcos;    
    }
    std::vector<ftype> oldmapsize(numdim_, 0);
    for (size_t i = 0; i < numdim_; ++i) { 
      oldmapsize[i] = (ftype) prev_layer->mapsize_[i];
    }
    ftype min0 = ((ftype) oldmapsize[0] / 2 - (ftype) 0.5) - (ftype) maxsize[0] / 2 - shift_[0];
    ftype max0 = ((ftype) oldmapsize[0] / 2 - (ftype) 0.5) + (ftype) maxsize[0] / 2 + shift_[0];
    ftype min1 = ((ftype) oldmapsize[1] / 2 - (ftype) 0.5) - (ftype) maxsize[1] / 2 - shift_[1];
    ftype max1 = ((ftype) oldmapsize[1] / 2 - (ftype) 0.5) + (ftype) maxsize[1] / 2 + shift_[1];
    if (!(0 <= min0 && max0 < oldmapsize[0] && 0 <= min1 && max1 < oldmapsize[1])) {
      mexPrintMsg("min1", min0); mexPrintMsg("max1", max0);
      mexPrintMsg("min2", min1); mexPrintMsg("max2", max1);    
      mexAssert(false, "For these jitter parameters the new image is out of the original image");
    }
  }
  if (mexIsField(mx_layer, "randtest")) {
    randtest_ = (mexGetScalar(mexGetField(mx_layer, "randtest")) > 0);    
  }
} 

void LayerJitt::Forward(Layer *prev_layer, int passnum) {
  
  if (passnum == 3) return;
  
  batchsize_ = prev_layer->batchsize_;  
  activ_mat_.resize(batchsize_, length_);

  if (passnum == 0) {
    if (mapsize_[0] == prev_layer->mapsize_[0] && 
        mapsize_[1] == prev_layer->mapsize_[1]) {
      activ_mat_ = prev_layer->activ_mat_;      
      return;
    }
    if (randtest_ == false) {
      // do just central cropping    
      shift_.assign(numdim_, 0);
      scale_.assign(numdim_, 1);
      mirror_.assign(numdim_, false);
      angle_ = 0;
    }    
  }

  #if COMP_REGIME != 2  
    std::vector< std::vector<Mat> > prev_activ, activ;    
    InitMaps(prev_layer->activ_mat_, prev_layer->mapsize_, prev_activ);
    InitMaps(activ_mat_, mapsize_, activ);    
    #if COMP_REGIME == 1
      #pragma omp parallel for
    #endif
    for (int k = 0; k < batchsize_; ++k) {  
      std::vector<ftype> shift(numdim_, 0);
      std::vector<ftype> scale(numdim_, 1);
      std::vector<bool> mirror(numdim_, false);
      ftype angle = 0;
      for (size_t j = 0; j < numdim_; ++j) {
        if (shift_[j] > 0) {
          shift[j] = ((ftype) std::rand() / RAND_MAX * 2 - 1) * shift_[j];          
        }
        if (scale_[j] > 1) {
          scale[j] = pow(scale_[j], (ftype) std::rand() / RAND_MAX * 2 - 1);          
        }
        if (mirror_[j]) {
          mirror[j] = ((ftype) std::rand() / RAND_MAX * 2 - 1 > 0);          
        }        
      }
      if (angle_ > 0) {
        angle = ((ftype) std::rand() / RAND_MAX * 2 - 1) * kPi * angle_;        
      }
      for (size_t i = 0; i < outputmaps_; ++i) {        
        Transform(prev_activ[k][i], shift, scale, mirror, angle, defval_, activ[k][i]);      
      }      
    }
  #else // GPU  
    TransformActs(prev_layer->activ_mat_, activ_mat_, prev_layer->mapsize_,
                  mapsize_, shift_, scale_, mirror_, angle_, defval_);
  #endif
  /*
  Mat immean(batchsize_, 1);
  Mean(prev_layer->activ_mat_, immean, 2);
  immean *= -1;
  activ_mat_.AddVect(immean, 2);
  */
  /*
  Mat pixmean(1, length_);
  Mean(activ_mat_, pixmean, 1);
  pixmean *= -1;
  activ_mat_.AddVect(pixmean, 1);
  */
  activ_mat_.Validate();  
  /*
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Jitt: activ_[0][0]", activ_[0][0](0, i)); 
  } */
}
