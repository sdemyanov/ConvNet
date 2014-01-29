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

#ifndef _NET_H_
#define _NET_H_

#include "layer.h"
#include "params.h"

class Net {

public:
  void InitLayers(const mxArray *mx_layers);
  void InitParams(const mxArray *mx_params);
  void Train(const mxArray *mx_data, const mxArray *mx_labels);
  void Classify(const mxArray *mx_data, mxArray *&mx_pred);
  void SetWeights(const mxArray *mx_weights);
  void GetWeights(mxArray *&mx_weights) const;
  void GetTrainError(mxArray *&mx_errors) const;  
  size_t NumWeights() const;  
  void Clear();
  
  Net() {};
  ~Net() { Clear(); }
  
private:
  std::vector<Layer*> layers_;
  Params params_;
  Mat data_, labels_, trainerror_;
  Mat classcoefs_; // in fact vector

  void ReadData(const mxArray *mx_data);
  void ReadLabels(const mxArray *mx_labels);
  void CalcDeriv(const Mat &labels_batch, ftype &loss);
  void Forward(Mat &data_batch, Mat &pred, bool istrain);
  void Backward(Mat &labels_batch, ftype &loss);
  void UpdateWeights(size_t epoch, bool isafter);  
};

#endif