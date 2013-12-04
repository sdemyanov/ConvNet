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

#include "layer_i.h"

LayerInput::LayerInput() {
  type_ = "i";    
}  
  
void LayerInput::Init(const mxArray *mx_layer, Layer *prev_layer) {
  
  mexAssert(prev_layer == NULL, "The 'i' type layer must be the first one");
  mexAssert(mexIsField(mx_layer, "mapsize"), "The first layer must contain the 'mapsize' field");
  std::vector<double> mapsize = mexGetVector(mexGetField(mx_layer, "mapsize"));
  mexAssert(mapsize.size() == 2, "Input mapsize must contain 2 values");
  mapsize_.resize(mapsize.size());  
  for (size_t i = 0; i < mapsize_.size(); ++i) {
    mapsize_[i] = (size_t) mapsize[i];        
  }  
  if (!mexIsField(mx_layer, "outputmaps")) {
    outputmaps_ = 1;
  } else {
    outputmaps_ = (size_t) mexGetScalar(mexGetField(mx_layer, "outputmaps"));
  }  
  activ_.resize(outputmaps_);
  deriv_.resize(outputmaps_);  
} 