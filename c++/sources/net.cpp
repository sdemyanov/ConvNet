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

#include "net.h"
#include "layer_i.h"
#include "layer_c.h"
#include "layer_s.h"
#include "layer_f.h"

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
  layers_.front()->Init(mx_layer, NULL); 
  for (size_t i = 1; i < layers_num; ++i) {    
    Layer *prev_layer = layers_[i-1];
    mx_layer = mexGetCell(mx_layers, i);  
    layer_type = mexGetString(mexGetField(mx_layer, "type"));
    if (layer_type == "c") {      
      layers_[i] = new LayerConv();
    } else if (layer_type == "s") {
      layers_[i] = new LayerScal();
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
  LayerFull *lastlayer = static_cast<LayerFull*>(layers_.back());
  std::vector<size_t> labels_dim = mexGetDimensions(mx_labels);  
  mexAssert(labels_dim.size() == 2, "The label array must have 2 dimensions");    
  mexAssert(labels_dim[0] == lastlayer->length_,
    "Labels and last layer must have equal number of classes");  
  size_t train_num = labels_dim[1];  
  Mat labels(labels_dim);
  mexGetMatrix(mx_labels, labels);
  if (lastlayer->function_ == "SVM") {
    (labels *= 2) -= 1;    
  }
  
  size_t mapnum = 1;  
  if (mexIsCell(mx_data)) {
    mapnum = mexGetNumel(mx_data);    
  }
  mexAssert(mapnum == layers_.front()->outputmaps_,
    "Data must have the same number of cells as outputmaps on the first layer");
  std::vector< std::vector<Mat> > data(mapnum);  
  for (size_t map = 0; map < mapnum; ++map) {
    const mxArray *mx_cell;  
    if (mexIsCell(mx_data)) {
      mx_cell = mxGetCell(mx_data, map);
    } else {
      mx_cell = mx_data;
    }
    std::vector<size_t> data_dim = mexGetDimensions(mx_cell);  
    mexAssert(data_dim.size() == 3, "The data array must have 3 dimensions");  
    mexAssert(data_dim[0] == layers_.front()->mapsize_[0] && 
              data_dim[1] == layers_.front()->mapsize_[1],
             "Data and the first layer must have equal sizes");    
    mexAssert(data_dim[2] == train_num, "All data maps and labels must have equal number of objects");    
    mexGetMatrix3D(mx_cell, data[map]);
  }
  
  classcoefs_.assign(labels_dim[0], 1);
  if (params_.balance_) {  
    Mat labels_mean(labels_dim[0], 1);
    labels.Mean(2, labels_mean);
    for (size_t i = 0; i < labels_dim[0]; ++i) {
      mexAssert(labels_mean(i) > 0, "Balancing impossible: one of the classes is not presented");  
      (classcoefs_[i] /= labels_mean(i)) /= labels_dim[0];      
    }
  }    
  
  size_t numbatches = ceil((double) train_num/params_.batchsize_);
  trainerror_.assign(params_.numepochs_ * numbatches, 0);
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
      std::vector< std::vector<Mat> > data_batch(mapnum);
      for (size_t map = 0; map < mapnum; ++map) {
        data_batch[map].resize(batchsize);
        for (size_t i = 0; i < batchsize; ++i) {        
          data_batch[map][i] = data[map][batch_ind[i]];
        }
      }      
      Mat labels_batch(labels_dim[0], batchsize);
      Mat pred_batch(labels_dim[0], batchsize);
      labels.SubMat(batch_ind, 2 ,labels_batch);
      UpdateWeights(false);      
      Forward(data_batch, pred_batch, true);
      Backward(labels_batch, trainerror_[epoch * numbatches + batch]);      
      UpdateWeights(true);
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

void Net::Classify(const mxArray *mx_data, Mat &pred) {
  
  //mexPrintMsg("Start classification...");  
  size_t mapnum = 1;  
  if (mexIsCell(mx_data)) {
    mapnum = mexGetNumel(mx_data);    
  }
  mexAssert(mapnum == layers_.front()->outputmaps_,
    "Data must have the same number of cells as outputmaps on the first layer");
  std::vector< std::vector<Mat> > data(mapnum);  
  for (size_t i = 0; i < mapnum; ++i) {
    const mxArray *mx_cell;  
    if (mexIsCell(mx_data)) {
      mx_cell = mxGetCell(mx_data, i);
    } else {
      mx_cell = mx_data;
    }
    std::vector<size_t> data_dim = mexGetDimensions(mx_cell);  
    mexAssert(data_dim.size() == 3, "The data array must have 3 dimensions");  
    mexAssert(data_dim[0] == layers_.front()->mapsize_[0] && 
              data_dim[1] == layers_.front()->mapsize_[1],
             "Data and the first layer must have equal sizes");      
    mexGetMatrix3D(mx_cell, data[i]);
  }

  Forward(data, pred, false);
  //mexPrintMsg("Classification finished");
}

void Net::Forward(const std::vector< std::vector<Mat> > &data_batch, Mat &pred, bool regime) {
  
  //mexPrintMsg("Start forward pass...");  
  layers_.front()->activ_ = data_batch;  
  layers_.front()->batchsize_ = layers_.front()->activ_[0].size();
  for (size_t i = 1; i < layers_.size(); ++i) {
    //mexPrintMsg("Forward pass for layer", i);
    layers_[i]->Forward(layers_[i-1], regime);
  }
  //mexPrintMsg("Forward pass finished");
  LayerFull *lastlayer = static_cast<LayerFull*>(layers_.back());  
  pred.resize(lastlayer->length_, lastlayer->batchsize_);
  pred = lastlayer->output_;  
}

void Net::Backward(const Mat &labels_batch, double &loss) {
  
  //mexPrintMsg("Start backward pass...");
  size_t batchsize = labels_batch.size2();
  size_t classes_num = labels_batch.size1();
  loss = 0;
  LayerFull *lastlayer = static_cast<LayerFull*>(layers_.back());
  mexAssert(batchsize == lastlayer->batchsize_, 
    "The number of objects in data and label batches is different");
  mexAssert(classes_num == lastlayer->length_, 
    "Labels in batch and last layer must have equal number of classes");  
  lastlayer->output_der_.init(classes_num, batchsize, 0);  
  
  if (lastlayer->function_ == "SVM") {
    Mat lossmat = lastlayer->output_;
    lossmat.ElemProd(labels_batch);
    (lossmat *= -1) += 1;    
    lossmat.ElemMax(0);
    lastlayer->output_der_ = lossmat;
    lastlayer->output_der_.ElemProd(labels_batch);
    lastlayer->output_der_ *= -2;
    //lastlayer->output_der_ *= curcoef;    
    // correct loss also contains weightsT * weights, but it is too long to calculate it
    lossmat.ElemProd(lossmat);
    loss = lossmat.Sum() / batchsize;    
  } else if (lastlayer->function_ == "sigmoid") {
    lastlayer->output_der_ = lastlayer->output_;
    lastlayer->output_der_ -= labels_batch;    
    Mat lossmat = lastlayer->output_der_;
    lossmat.ElemProd(lastlayer->output_der_);
    loss = lossmat.Sum() / (2 * batchsize);
  } else {
    mexAssert(false, "Net::Backward");
  }
  lastlayer->output_der_.MultVect(classcoefs_, 1);  
  
  for (size_t i = layers_.size() - 1; i > 0; --i) {
    //mexPrintMsg("Backward pass for layer", i);    
    layers_[i]->Backward(layers_[i-1]);
  }  
  //mexPrintMsg("Backward pass finished");  
}

void Net::UpdateWeights(bool isafter) {
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->UpdateWeights(params_, isafter);
  }  
}

std::vector<double> Net::GetWeights() const {
  std::vector<double> weights;  
  for (size_t i = 1; i < layers_.size(); ++i) {    
    layers_[i]->GetWeights(weights);
  }  
  return weights;
}  

void Net::SetWeights(std::vector<double> &weights) {  
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->SetWeights(weights);
  }
  mexAssert(weights.size() == 0, "Vector of weights is too long!");
}

std::vector<double> Net::GetTrainError() const {
  return trainerror_;
}

Net::~Net() {
  for (size_t i = 0; i < layers_.size(); ++i){
    delete layers_[i];
  }
}