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

#include "net.h"

#define NARGIN 3
#define IN_L pRhs[0] // layers
#define IN_W pRhs[1] // weights
#define IN_X pRhs[2] // data

#define NARGOUT 1
#define OUT_P	pLhs[0] // predictions

int print = 1;

void mexFunction(int nLhs, mxArray* pLhs[], int nRhs, const mxArray* pRhs[]) {

  mexAssert(nRhs == NARGIN, "Number of input arguments is not correct!" );
  mexAssert(nLhs == NARGOUT, "Number of output arguments is wrong!" );  
  mexAssert(mexIsCell(IN_L), "Layers must be the cell array");
  mexAssert(mexGetNumel(IN_L) == 2, "Layers array must contain 2 cells");
  mexAssert(mexIsCell(IN_W), "Weights must be the cell array");
  mexAssert(mexGetNumel(IN_W) == 2, "Weights array must contain 2 cells");
  
  Net net, imnet;
  net.InitLayers(mexGetCell(IN_L, 1));
  net.InitWeights(mexGetCell(IN_W, 1));  
  imnet.InitLayers(mexGetCell(IN_L, 0));        
  imnet.ReadData(IN_X);
  
  const mxArray *mx_imweights = mexGetCell(IN_W, 0);  
  size_t train_num = mexGetNumel(mx_imweights);
  size_t pixels_num = imnet.data_.size1();
  Layer *firstlayer = net.layers_[0];  
  mexAssert(pixels_num == firstlayer->outputmaps_ * 
            firstlayer->mapsize_[0] * firstlayer->mapsize_[1],
            "Pixels number must coincide with the first layer elements number");   
  
  std::vector<size_t> pred_size(2);
  pred_size[0] = 1; pred_size[1] = pixels_num;    
  Mat images_mat(train_num, pixels_num);
  std::vector< std::vector<Mat> > images;
  InitMaps(images_mat, pred_size, images);
  Mat pred_mat, pred_pixels; 
  
  imnet.InitActiv(imnet.data_);
  for (size_t m = 0; m < train_num; ++m) {
    imnet.InitWeights(mexGetCell(mx_imweights, m));    
    imnet.Forward(pred_pixels, 1);        
    images[m][0].copy(Trans(pred_pixels));    
  } 
  net.InitActiv(images_mat);
  net.Forward(pred_mat, 0);
  OUT_P = mexSetMatrix(pred_mat);  
}
