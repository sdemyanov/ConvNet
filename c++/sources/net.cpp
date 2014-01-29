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
#include "layer_i.h"
#include "layer_j.h"
#include "layer_c.h"
#include "layer_s.h"
#include "layer_t.h"
#include "layer_f.h"
#include <ctime>

void Net::InitLayers(const mxArray *mx_layers) {
  
  //mexPrintMsg("Start layers initialization...");
  std::srand((unsigned) std::time(0));  
  size_t layers_num = mexGetNumel(mx_layers);  
  mexAssert(layers_num >= 2, "The net must contain at least 2 layers");
  const mxArray *mx_layer = mexGetCell(mx_layers, 0);  
  std::string layer_type = mexGetString(mexGetField(mx_layer, "type"));   
  mexAssert(layer_type == "i", "The first layer must be the type of 'i'");
  layers_.resize(layers_num);
  layers_[0] = new LayerInput();
  //mexPrintMsg("Initializing layer of type", layer_type);    
  layers_[0]->Init(mx_layer, NULL); 
  for (size_t i = 1; i < layers_num; ++i) {    
    Layer *prev_layer = layers_[i-1];
    mx_layer = mexGetCell(mx_layers, i);  
    layer_type = mexGetString(mexGetField(mx_layer, "type"));
    if (layer_type == "j") {
      layers_[i] = new LayerJitter();
    } else if (layer_type == "c") {      
      layers_[i] = new LayerConv();
    } else if (layer_type == "s") {
      layers_[i] = new LayerScal();
    } else if (layer_type == "t") {
      layers_[i] = new LayerTrim();
    } else if (layer_type == "f") {
      layers_[i] = new LayerFull();
    } else {
      mexAssert(false, layer_type + " - unknown type of the layer");
    }    
    //mexPrintMsg("Initializing layer of type", layer_type);    
    layers_[i]->Init(mx_layer, prev_layer);    
  }
  mexAssert(layer_type == "f", "The last layer must be the type of 'f'");
  //mexPrintMsg("Layers initialization finished");
}

void Net::InitParams(const mxArray *mx_params) {
  //mexPrintMsg("Start params initialization...");
  params_.Init(mx_params);
  //mexPrintMsg("Params initialization finished");
}

void Net::Train(const mxArray *mx_data, const mxArray *mx_labels) {  
  
  //mexPrintMsg("Start training...");  
  ReadData(mx_data);
  ReadLabels(mx_labels);
  
  size_t train_num = labels_.size1();
  size_t classes_num = labels_.size2();
  size_t numbatches = (size_t) ceil((ftype) train_num/params_.batchsize_);
  trainerror_.resize(params_.numepochs_, numbatches);
  Mat data_batch(params_.batchsize_, data_.size2());
  Mat labels_batch(params_.batchsize_, classes_num);
  Mat pred_batch;
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
      SubMat(data_, batch_ind, 1, data_batch);      
      SubMat(labels_, batch_ind, 1, labels_batch);      
      UpdateWeights(epoch, false);      
      Forward(data_batch, pred_batch, true);
      Backward(labels_batch, trainerror_(epoch, batch));      
      UpdateWeights(epoch, true);
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
}

void Net::Classify(const mxArray *mx_data, mxArray *&mx_pred) {  
  //mexPrintMsg("Start classification...");
  ReadData(mx_data);
  Mat pred;
  Forward(data_, pred, false);
  mx_pred = mexSetMatrix(pred);
  //mexPrintMsg("Classification finished");
}

void Net::Forward(Mat &data_batch, Mat &pred, bool istrain) {
  
  mexAssert(layers_.size() >= 2 , "The net is not initialized");
  //mexPrintMsg("Start forward pass...");
  layers_[0]->activ_mat_.attach(data_batch);
  //mexPrintMsg("Forward pass for layer", layers_[0]->type_);
  layers_[0]->Forward(NULL, istrain);
  for (size_t i = 1; i < layers_.size(); ++i) {
    //mexPrintMsg("Forward pass for layer", layers_[i]->type_);
    layers_[i]->Forward(layers_[i-1], istrain);
    if (utIsInterruptPending()) {
      Clear();
      mexAssert(false, "Ctrl-C Detected. END");
    }
  }  
  pred.attach(layers_.back()->activ_mat_);  
  //("Forward pass finished");
}

void Net::Backward(Mat &labels_batch, ftype &loss) {
  
  //mexPrintMsg("Start backward pass...");
  CalcDeriv(labels_batch, loss);
  for (size_t i = layers_.size() - 1; i > 0; --i) {
    //mexPrintMsg("Backward pass for layer", layers_[i]->type_);    
    layers_[i]->Backward(layers_[i-1]);
  }
  //mexPrintMsg("Backward pass for layer", layers_[0]->type_);
  layers_[0]->Backward(NULL);
  //mexPrintMsg("Backward pass finished");  
}

void Net::UpdateWeights(size_t epoch, bool isafter) {
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->UpdateWeights(params_, epoch, isafter);
  }
}

void Net::ReadData(const mxArray *mx_data) {
  std::vector<size_t> data_dim = mexGetDimensions(mx_data);
  mexAssert(data_dim.size() == 4, "The data array must have 4 dimensions");  
  mexAssert(data_dim[0] == layers_[0]->mapsize_[0] && 
            data_dim[1] == layers_[0]->mapsize_[1],
    "Data and the first layer must have equal sizes");  
  mexAssert(data_dim[2] == layers_[0]->outputmaps_,
    "Data's 3rd dimension must be equal to the outputmaps on the input layer");
  mexAssert(data_dim[3] > 0, "Input data array is empty");
  ftype *data = mexGetPointer(mx_data);
  data_.attach(data, data_dim[3], data_dim[0] * data_dim[1] * data_dim[2]);  
}

void Net::ReadLabels(const mxArray *mx_labels) {
  std::vector<size_t> labels_dim = mexGetDimensions(mx_labels);  
  mexAssert(labels_dim.size() == 2, "The label array must have 2 dimensions");
  //mexPrintMsg("labels_dim.", labels_dim[0]);
  //mexPrintMsg("data_.size()", data_.size());  
  mexAssert(labels_dim[0] == data_.size1(),
    "All data maps and labels must have equal number of objects");
  size_t classes_num = labels_dim[1];
  mexAssert(classes_num == layers_.back()->length_,
    "Labels and last layer must have equal number of classes");  
  mexGetMatrix(mx_labels, labels_);
  classcoefs_.init(1, classes_num, 1);
  if (params_.balance_) {  
    Mat labels_mean = Mean(labels_, 1);
    for (size_t i = 0; i < classes_num; ++i) {
      mexAssert(labels_mean(i) > 0, "Balancing impossible: one of the classes is not presented");
      (classcoefs_(i) /= labels_mean(i)) /= classes_num;      
    }
  }
  if (layers_.back()->function_ == "SVM") {
    (labels_ *= 2) -= 1;    
  }
}

void Net::CalcDeriv(const Mat &labels_batch, ftype &loss) {  
  size_t batchsize = labels_batch.size1();
  size_t classes_num = labels_batch.size2();
  Layer *lastlayer = layers_.back();
  mexAssert(batchsize == lastlayer->batchsize_, 
    "The number of objects in data and label batches is different");
  mexAssert(classes_num == lastlayer->length_, 
    "Labels in batch and last layer must have equal number of classes");  
  if (lastlayer->function_ == "SVM") {
    Mat lossmat = lastlayer->activ_mat_;
    (((lossmat *= labels_batch) *= -1) += 1).ElemMax(0);        
    lastlayer->deriv_mat_ = lossmat;
    (lastlayer->deriv_mat_ *= labels_batch) *= -2;    
    // correct loss also contains weightsT * weights / C, but it is too long to calculate it
    loss = (lossmat *= lossmat).Sum() / batchsize;    
  } else if (lastlayer->function_ == "sigmoid") {
    lastlayer->deriv_mat_ = lastlayer->activ_mat_;
    lastlayer->deriv_mat_ -= labels_batch;    
    Mat lossmat = lastlayer->deriv_mat_;
    loss = (lossmat *= lastlayer->deriv_mat_).Sum() / (2 * batchsize);
  } else {
    mexAssert(false, "Net::Backward");
  }
  lastlayer->deriv_mat_.MultVect(classcoefs_, 2);  
}

size_t Net::NumWeights() const {
  size_t num_weights = 0;
  for (size_t i = 1; i < layers_.size(); ++i) {    
    num_weights += layers_[i]->NumWeights();
  }
  return num_weights;
}

void Net::GetWeights(mxArray *&mx_weights) const {
  size_t num_weights = NumWeights();
  //mexPrintMsg("num_weights", num_weights);
  mx_weights = mexNewMatrix(1, num_weights);  
  ftype *weights = mexGetPointer(mx_weights);
  ftype *weights_end = weights + num_weights;
  //mexPrintMsg("weights_end - weights", weights_end - weights);
  for (size_t i = 1; i < layers_.size(); ++i) {    
    layers_[i]->GetWeights(weights, weights_end);
  }
  //mexPrintMsg("weights_end - weights", weights_end - weights);
  mexAssert(weights == weights_end, "In GetWeights the vector of weights is too long!");
}  

void Net::SetWeights(const mxArray *mx_weights) {
  ftype *weights, *weights_end;
  if (mx_weights == NULL) {
    weights = NULL;
    weights_end = NULL;    
  } else {
    size_t num_weights = mexGetNumel(mx_weights);
    weights = mexGetPointer(mx_weights);  
    weights_end = weights + num_weights;    
  }
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->SetWeights(weights, weights_end);
  }
  mexAssert(weights == weights_end, "In SetWeights the vector of weights is too long!");
}

void Net::GetTrainError(mxArray *&mx_errors) const {  
  mx_errors = mexSetMatrix(trainerror_);
}

void Net::Clear() {
  for (size_t i = 0; i < layers_.size(); ++i){
    delete layers_[i];
  }
  layers_.clear();
  data_.clear();
  labels_.clear();
  trainerror_.clear();
  classcoefs_.clear();
}
