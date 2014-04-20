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

#define NARGIN_MIN 4
#define NARGIN_MAX 5
#define IN_L pRhs[0] // layers
#define IN_W pRhs[1] // weights
#define IN_X pRhs[2] // data
#define IN_Y pRhs[3] // labels
#define IN_P pRhs[4] // params

#define NARGOUT 2
#define OUT_W	pLhs[0] // weights
#define OUT_E pLhs[1] // train errors on each batch

int print = 1;

void mexFunction(int nLhs, mxArray* pLhs[], int nRhs, const mxArray* pRhs[]) {

  mexAssert(NARGIN_MIN <= nRhs && nRhs <= NARGIN_MAX, "Number of input arguments in wrong!");
  mexAssert(nLhs == NARGOUT, "Number of output arguments is wrong!" );  
  mexAssert(mexIsCell(IN_L), "Layers must be the cell array");
  mexAssert(mexGetNumel(IN_L) == 2, "Layers array must contain 2 cells");
  mexAssert(mexIsCell(IN_W), "Weights must be the cell array");
  mexAssert(mexGetNumel(IN_W) == 2, "Weights array must contain 2 cells");
  
  Net net;
  mxArray *mx_weights;
  net.InitLayers(mexGetCell(IN_L, 1));
  net.InitWeights(mexGetCell(IN_W, 1), mx_weights);  
  net.InitParams(IN_P);
  net.ReadLabels(IN_Y);

  const mxArray *mx_imweights = mexGetCell(IN_W, 0);  
  size_t train_num = net.labels_.size1();
  mexAssert(train_num == mexGetNumel(mx_imweights),
    "Weights and labels number must coincide");
  bool is_multicoords = false;
  if (mexIsCell(IN_X)) {    
    mexAssert(train_num == mexGetNumel(IN_X),
    "Coordinates and labels number must coincide");
    is_multicoords = true;
  }
  Params params_ = net.params_;
  size_t numbatches = (size_t) ceil((ftype) train_num/params_.batchsize_);  
  Mat trainerror_(params_.numepochs_, numbatches);
  Mat trainerror2_(params_.numepochs_, numbatches);
  trainerror2_.assign(0);
  
  std::vector<Net> imnets;
  imnets.resize(params_.batchsize_);
  for (size_t i = 0; i < params_.batchsize_; ++i) {
    imnets[i].InitLayers(mexGetCell(IN_L, 0));        
    if (!is_multicoords) {
      imnets[i].ReadData(IN_X);    
    } else {
      imnets[i].ReadData(mexGetCell(IN_X, i)); // just to get pixels_num
    }
  }
  size_t pixels_num = imnets[0].data_.size1();
  Layer *firstlayer = net.layers_[0];
  size_t dimens_num = firstlayer->outputmaps_;
  mexAssert(imnets[0].layers_.back()->length_ == dimens_num,
            "Final layer length must coincide with the number of outputmaps");   
  mexAssert(pixels_num == firstlayer->mapsize_[0] * firstlayer->mapsize_[1],
            "Pixels number must coincide with the first layer elements number");   
  
  std::vector<size_t> pred_size(2);
  pred_size[0] = 1; pred_size[1] = pixels_num * dimens_num;    
  Mat images_mat, labels_batch, pred_batch, pred_pixels;      
  std::vector< std::vector<Mat> > images, images_der;
  for (size_t epoch = 0; epoch < params_.numepochs_; ++epoch) {    
    std::vector<size_t> randind(train_num);
    for (size_t i = 0; i < train_num; ++i) {
      randind[i] = i;
    }
    if (params_.shuffle_) {
      std::random_shuffle(randind.begin(), randind.end());
    }
    std::vector<size_t>::const_iterator iter = randind.begin();
    for (size_t batch = 0; batch < numbatches; ++batch) {
      size_t batchsize = std::min(params_.batchsize_, (size_t)(randind.end() - iter));
      std::vector<size_t> batch_ind = std::vector<size_t>(iter, iter + batchsize);
      iter = iter + batchsize;      
      labels_batch = SubMat(net.labels_, batch_ind, 1);
      net.UpdateWeights(epoch, false);
      images_mat.resize(batchsize, pred_size[1]);
      InitMaps(images_mat, pred_size, images);        
      // first pass
      for (size_t m = 0; m < batchsize; ++m) {        
        imnets[m].InitWeights(mexGetCell(mx_imweights, batch_ind[m]));
        if (is_multicoords) {
          imnets[m].ReadData(mexGetCell(IN_X, batch_ind[m]));
        }
        imnets[m].InitActiv(imnets[m].data_);                
        imnets[m].Forward(pred_pixels, 1);        
        images[m][0].copy(Trans(pred_pixels).reshape(pred_size[0], pred_size[1]));
      }
      net.InitActiv(images_mat);  
      net.Forward(pred_batch, 1);
      /*
      for (int i = 0; i < 5; ++i) {
        mexPrintMsg("pred_batch1", pred_batch(0, i)); 
      }*/
      // second pass
      net.InitDeriv(labels_batch, trainerror_(epoch, batch));
      net.Backward();
      net.CalcWeights();
      InitMaps(firstlayer->deriv_mat_, pred_size, images_der);
      
      for (size_t m = 0; m < batchsize; ++m) {
        imnets[m].layers_.back()->deriv_mat_ = Trans(images_der[m][0].reshape(dimens_num, pixels_num));        
        imnets[m].Backward();        
      }
      
      // third pass      
      ftype loss2 = 0, curloss = 0, invind = 0; 
      std::vector<size_t> invalid;
      for (size_t m = 0; m < batchsize; ++m) {        
        imnets[m].InitDeriv2(curloss);
        if (curloss > 0) {
          imnets[m].Forward(pred_pixels, 3);        
          images[m][0].copy(Trans(pred_pixels).reshape(pred_size[0], pred_size[1]));        
          loss2 += curloss;
        } else {
          invalid.push_back(m);          
        }
      }
      if (invalid.size() < batchsize) {
        loss2 /= (batchsize - invalid.size());
        trainerror2_(epoch, batch) = loss2;
        net.InitActiv(images_mat);  
        net.Forward(pred_batch, 3);
      }        
      net.CalcWeights2(invalid);      
      // weights update      
      net.UpdateWeights(epoch, true);      
      if (params_.verbose_ == 2) {
        std::string info = std::string("Epoch: ") + std::to_string(epoch+1) +
                           std::string(", batch: ") + std::to_string(batch+1);
        mexPrintMsg(info);
      }      
    } // batch    
    if (params_.verbose_ == 1) {
      std::string info = std::string("Epoch: ") + std::to_string(epoch+1);
      mexPrintMsg(info);
    }
  } // epoch
  //mexPrintMsg("Training finished");
  
  //net.weights_.get().copy(net.weights_.der());
    
  OUT_W = mexSetCellMat(1, 2);  
  mexSetCell(OUT_W, 0, mexDuplicateArray(mx_imweights));
  mexSetCell(OUT_W, 1, mx_weights);
  
  OUT_E = mexSetCellMat(1, 2);  
  mexSetCell(OUT_E, 0, mexSetMatrix(trainerror_));  
  mexSetCell(OUT_E, 1, mexSetMatrix(trainerror2_));
}
