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

#include "layer_jitt.h"

LayerJitt::LayerJitt() {
  function_ = "none";
  add_bias_ = false;
  shift_.resize(1, 2);
  shift_.assign(0);
  scale_.resize(1, 2);
  scale_.assign(1);
  mirror_.resize(1, 2);
  mirror_.assign(0);
  angle_ = 0;
  defval_ = 0;
  noise_std_ = 0;
  randtest_ = false;
}

void LayerJitt::Init(const mxArray *mx_layer, const Layer *prev_layer) {
  std::vector<ftype> shift(2);
  shift[0] = 0; shift[1] = 0;
  if (mexIsField(mx_layer, "shift")) {
    shift = mexGetVector(mexGetField(mx_layer, "shift"));
    mexAssertMsg(shift.size() == 2, "Length of jitter shift vector and maps dimensionality must coincide");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(0 <= shift[i] && shift[i] < dims_[i+2], "Shift in 'jitt' layer is out of range");
    }
    MatCPU shift_cpu(1, 2);
    shift_cpu.assign(shift);
    shift_ = shift_cpu;
  }

  std::vector<ftype> scale(2);
  scale[0] = 1; scale[1] = 1;
  if (mexIsField(mx_layer, "scale")) {
    scale = mexGetVector(mexGetField(mx_layer, "scale"));
    mexAssertMsg(scale.size() == 2, "Length of jitter scale vector and maps dimensionality must coincide");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(1 <= scale[i] && scale[i] < dims_[i+2], "Scale in 'j' layer is out of range");
    }
    MatCPU scale_cpu(1, 2);
    scale_cpu.assign(scale);
    scale_ = scale_cpu;
    scale_.Log();
  }

  if (mexIsField(mx_layer, "mirror")) {
    std::vector<ftype> mirror = mexGetVector(mexGetField(mx_layer, "mirror"));
    mexAssertMsg(mirror.size() == 2, "Length of jitter scale vector and maps dimensionality must coincide");
    for (size_t i = 0; i < 2; ++i) {
      mexAssertMsg(mirror[i] == 0 || mirror[i] == 1, "Mirror must be either 0 or 1");
    }
    MatCPU mirror_cpu(1, 2);
    mirror_cpu.assign(mirror);
    mirror_ = mirror_cpu;
  }

  if (mexIsField(mx_layer, "angle")) {
    angle_ = mexGetScalar(mexGetField(mx_layer, "angle"));
    mexAssertMsg(0 <= angle_ && angle_ <= 1, "Angle in 'j' layer must be between 0 and 1");
  }

  if (mexIsField(mx_layer, "defval")) {
    defval_ = mexGetScalar(mexGetField(mx_layer, "defval"));
  } else {
    // check that the transformed image is always inside the original one
    std::vector<ftype> maxsize(2, 0);
    for (size_t i = 0; i < 2; ++i) {
      maxsize[i] = (ftype) (dims_[i+2] - 1) * scale[i];
    }
    if (angle_ > 0) {
      ftype angle_inn = atan2((ftype) dims_[2], (ftype) dims_[3]) / kPi;
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
    std::vector<ftype> oldmapsize(2, 0);
    for (size_t i = 0; i < 2; ++i) {
      oldmapsize[i] = (ftype) prev_layer->dims_[i+2];
    }
    ftype min0 = ((ftype) oldmapsize[0] / 2 - (ftype) 0.5) - (ftype) maxsize[0] / 2 - shift[0];
    ftype max0 = ((ftype) oldmapsize[0] / 2 - (ftype) 0.5) + (ftype) maxsize[0] / 2 + shift[0];
    ftype min1 = ((ftype) oldmapsize[1] / 2 - (ftype) 0.5) - (ftype) maxsize[1] / 2 - shift[1];
    ftype max1 = ((ftype) oldmapsize[1] / 2 - (ftype) 0.5) + (ftype) maxsize[1] / 2 + shift[1];
    if (!(0 <= min0 && max0 < oldmapsize[0] && 0 <= min1 && max1 < oldmapsize[1])) {
      mexPrintMsg("min1", min0); mexPrintMsg("max1", max0);
      mexPrintMsg("min2", min1); mexPrintMsg("max2", max1);
      mexAssertMsg(false, "For these jitter parameters the new image is out of the original image");
    }
  }
  if (mexIsField(mx_layer, "eigenvectors")) {
    const mxArray* mx_ev = mexGetField(mx_layer, "eigenvectors");
    std::vector<size_t> ev_dim = mexGetDimensions(mx_ev);
    mexAssertMsg(ev_dim.size() == 2, "The eigenvectors array must have 2 dimensions");
    mexAssertMsg(ev_dim[0] == dims_[1] && ev_dim[1] == dims_[1],
      "The eigenvector matrix size is wrong");
    MatCPU ev_cpu(dims_[1], dims_[1]);
    mexGetMatrix(mx_ev, ev_cpu);
    eigenvectors_.resize(dims_[1], dims_[1]);
    eigenvectors_ = ev_cpu;
    if (mexIsField(mx_layer, "noise_std")) {
      noise_std_ = mexGetScalar(mexGetField(mx_layer, "noise_std"));
      mexAssertMsg(noise_std_ >= 0, "noise_std must be nonnegative");
    } else {
      mexAssertMsg(false, "noise_std is required with eigenvalues");
    }
  }
  if (mexIsField(mx_layer, "randtest")) {
    randtest_ = (mexGetScalar(mexGetField(mx_layer, "randtest")) > 0);
  }
}

void LayerJitt::TransformForward(Layer *prev_layer, PassNum passnum) {

  shift_mat_.resize(dims_[0], 2);
  scale_mat_.resize(dims_[0], 2);
  mirror_mat_.resize(dims_[0], 2);
  angle_mat_.resize(dims_[0], 1);

  if (passnum == PassNum::ForwardTest) {
    if (dims_[2] == prev_layer->dims_[2] &&
        dims_[3] == prev_layer->dims_[3]) {
      activ_mat_ = prev_layer->activ_mat_;
      return;
    }
    shift_mat_.assign(0);
    scale_mat_.assign(1);
    mirror_mat_.assign(0);
    angle_mat_.assign(0);
  } else if (passnum == PassNum::Forward) {
    ((shift_mat_.rand() *= 2) -= 1).MultVect(shift_, 1);
    (((scale_mat_.rand() *= 2) -= 1).MultVect(scale_, 1)).Exp();
    (mirror_mat_.rand()).MultVect(mirror_, 1);
    ((angle_mat_.rand() *= 2) -= 1) *= (kPi * angle_);
  } else {
    // if ForwardLinear, use the existing ones
  }
  AffineTransform(prev_layer->activ_mat_, activ_mat_,
                  shift_mat_, scale_mat_, mirror_mat_, angle_mat_,
                  defval_, true);
  /*
  if (noise_std_ > 0 && passnum != PassNum::ForwardTest) {
    VaryColors(activ_mat_, dims_, eigenvectors_, noise_std_);
  } */
  activ_mat_.Validate();
}

void LayerJitt::TransformBackward(Layer *prev_layer) {
  // spaces outside fill in with zeros
  AffineTransform(deriv_mat_, prev_layer->deriv_mat_,
                shift_mat_, scale_mat_, mirror_mat_, angle_mat_,
                0, false);
}

