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

#ifndef _NET_H_
#define _NET_H_

#include "layer.h"
#include "params.h"
#include <vector>

class Net {

public:
  void InitLayers(const mxArray *mx_layers);
  void InitParams(const mxArray *mx_params);
  void Train(const mxArray *mx_data, const mxArray *mx_labels);
  void Classify(const mxArray *mx_data, Mat &pred);
  void SetWeights(std::vector<double> &weights);
  std::vector<double> GetWeights() const;
  std::vector<double> GetTrainError() const;
  
  Net() {};
  ~Net();  
  
private:
  std::vector<Layer*> layers_;
  Params params_;
  std::vector<double> classcoefs_;
  std::vector<double> trainerror_;  

  void Forward(const std::vector< std::vector<Mat> > &data_batch, Mat &pred, bool istrain);
  void Backward(const Mat &labels_batch, double &loss);
  void UpdateWeights(bool isafter);

};

#endif