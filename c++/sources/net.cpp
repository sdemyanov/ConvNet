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
#include "layer_n.h"
#include "layer_s.h"
#include "layer_f.h"

Net::Net() {
  //mexAssert(kDefaultOrder == false, "kDefaultOrder should be false");  
  mexAssert(kMapsOrder == false, "kMapsOrder should be false");  
  #if COMP_REGIME == 2 // GPU        
    mexAssert(PRECISION == 1, "In the GPU version PRECISION should be 1");    
    MatGPU::CudaInit();
  #endif
}

void Net::InitRand(size_t seed) {
  std::srand((unsigned int) seed);
  MatCPU::InitRand(seed);
  #if COMP_REGIME == 2 // GPU    
    MatGPU::InitRand(seed);
  #endif
}

void Net::InitLayers(const mxArray *mx_layers) {

  //mexPrintMsg("Start layers initialization...");  
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
      layers_[i] = new LayerJitt();
    } else if (layer_type == "c") {      
      layers_[i] = new LayerConv();
    } else if (layer_type == "n") {
      layers_[i] = new LayerNorm();
    } else if (layer_type == "s") {
      layers_[i] = new LayerScal();
    } else if (layer_type == "f") {
      layers_[i] = new LayerFull();
    } else {
      mexAssert(false, layer_type + " - unknown type of the layer");
    }    
    //mexPrintMsg("Initializing layer of type", layer_type);    
    layers_[i]->Init(mx_layer, prev_layer);    
    mexAssert(layers_[i]->function_ != "soft" || i == layers_num - 1,
              "Softmax function may be only on the last layer");
  }
  mexAssert(layer_type == "f", "The last layer must be the type of 'f'");
  LayerFull *lastlayer = static_cast<LayerFull*>(layers_.back());
  mexAssert(lastlayer->function_ == "soft" || lastlayer->function_ == "sigm",
            "The last layer function must be either 'soft' or 'sigm'");
  mexAssert(lastlayer->dropout_ == 0, "The last layer dropout must be 0");
  //mexPrintMsg("Layers initialization finished");
}

void Net::InitParams(const mxArray *mx_params) {
  //mexPrintMsg("Start params initialization...");
  params_.Init(mx_params);
  InitRand(params_.seed_);
  //mexPrintMsg("Params initialization finished");
}

void Net::Train(const mxArray *mx_data, const mxArray *mx_labels) {  

  //mexPrintMsg("Start training...");  
  ReadData(mx_data);
  ReadLabels(mx_labels);
  InitNorm();
  
  size_t train_num = data_.size1();
  size_t numbatches = DIVUP(train_num, params_.batchsize_);
  trainerrors_.resize(params_.epochs_, 2);
  trainerrors_.assign(0);
  Mat data_batch, labels_batch, pred_batch;
  for (size_t epoch = 0; epoch < params_.epochs_; ++epoch) {
    ftype beta;
    if (params_.beta_.size() == 1) {
      beta = params_.beta_[0];
    } else {
      beta = params_.beta_[epoch];
    }
    //print = 1;
    if (params_.shuffle_) {
      Shuffle(data_, labels_);      
    }
    StartTimer();
    //MatGPU::StartCudaTimer();
    size_t offset = 0;
    for (size_t batch = 0; batch < numbatches; ++batch) {        
      size_t batchsize = MIN(train_num - offset, params_.batchsize_);      
      UpdateWeights(epoch, false);
      data_batch.resize(batchsize, data_.size2());
      labels_batch.resize(batchsize, labels_.size2());      
      SubSet(data_, data_batch, offset, true);      
      SubSet(labels_, labels_batch, offset, true);
      ftype error1, error2;
      InitActiv(data_batch);
      Forward(pred_batch, 1);            
      InitDeriv(labels_batch, error1);
      trainerrors_(epoch, 0) += error1;
      Backward();
      InitDeriv2(error2);
      trainerrors_(epoch, 1) += error2;
      if (beta > 0) {
        Forward(pred_batch, 3);
      }
      UpdateWeights(epoch, true); 
      offset += batchsize;
      if (params_.verbose_ == 2) {
        mexPrintInt("Epoch", (int) epoch + 1);
        mexPrintInt("Batch", (int) batch + 1);
      }
    } // batch       
    //MatGPU::MeasureCudaTime("totaltime");
    MeasureTime("totaltime");
    if (params_.verbose_ == 1) {
      mexPrintInt("Epoch", (int) epoch + 1);
    }        
  } // epoch  
  trainerrors_ /= (ftype) numbatches;
  //mexPrintMsg("Training finished");
}

void Net::Classify(const mxArray *mx_data, mxArray *&mx_pred) {  

  //mexPrintMsg("Start classification...");
  ReadData(mx_data);
  size_t test_num = data_.size1();
  labels_.resize(test_num, layers_.back()->length_);  
  labels_.reorder(true, false);
  MatCPU curlabels;
  if (params_.test_epochs_ > 1) {
    curlabels.resize(test_num, layers_.back()->length_);
    curlabels.reorder(true, false);
    labels_.assign(0);
  } else {
    curlabels.attach(labels_);
  }
  size_t numbatches = DIVUP(test_num, params_.batchsize_);
  Mat data_batch, pred_batch;    
  for (size_t epoch = 0; epoch < params_.test_epochs_; ++epoch) {
    size_t offset = 0;
    for (size_t batch = 0; batch < numbatches; ++batch) {
      size_t batchsize = MIN(test_num - offset, params_.batchsize_);
      data_batch.resize(batchsize, data_.size2());
      SubSet(data_, data_batch, offset, true);    
      InitActiv(data_batch);
      Forward(pred_batch, 0);            
      SubSet(curlabels, pred_batch, offset, false);    
      offset += batchsize;
      if (params_.verbose_ == 2) {
        mexPrintInt("Test epoch", (int) epoch + 1);
        mexPrintInt("Test batch", (int) batch + 1);
      }      
    }
    if (params_.test_epochs_ > 1) {
      labels_ += curlabels;
    }
    if (params_.verbose_ == 1) {
      mexPrintInt("Test epoch", (int) epoch + 1);
    }
  }
  if (params_.test_epochs_ > 1) {
    labels_ /= (ftype) params_.test_epochs_;
  }
  labels_.reorder(kMatlabOrder, true);
  mx_pred = mexSetMatrix(labels_);
  //mexPrintMsg("Classification finished");
}

void Net::InitNorm() {

  LayerInput *firstlayer = static_cast<LayerInput*>(layers_[0]);
  size_t num_weights = firstlayer->NumWeights();
  if (num_weights == 0) return;
  MatCPU data = data_; // to reorder
  MatCPU mean_vect(1, data.size2());
  Mean(data, mean_vect, 1);
  mean_vect *= -1;
  if (firstlayer->is_mean_) {
    firstlayer->mean_weights_.get() = mean_vect;
    firstlayer->mean_weights_.get() += firstlayer->mean_;
  }
  if (firstlayer->is_maxdev_) {
    data *= data;
    MatCPU stdev_vect(1, data.size2());
    Mean(data, stdev_vect, 1);
    stdev_vect.Sqrt();
    stdev_vect.CondAssign(stdev_vect, false, firstlayer->maxdev_, firstlayer->maxdev_);        
    MatCPU maxdev_weights(1, data.size2());
    maxdev_weights.assign(firstlayer->maxdev_) /= stdev_vect;        
    firstlayer->maxdev_weights_.get() = maxdev_weights;    
  }
}

void Net::InitActiv(const Mat &data) {
  mexAssert(layers_.size() >= 2 , "The net is not initialized");
  first_layer_ = 0;
  layers_[0]->activ_mat_.attach(data);
  layers_[0]->activ_mat_.Validate();
}

void Net::Forward(Mat &pred, int passnum) {
  if (first_layer_ == 0) {
    //mexPrintMsg("Forward pass for layer", layers_[0]->type_);  
    layers_[0]->Forward(NULL, passnum);  
    layers_[0]->CalcWeights(NULL, passnum);
  }
  for (size_t i = first_layer_; i < layers_.size(); ++i) {
    if (layers_[i]->type_ == "j") first_layer_ = i;
    //mexPrintMsg("Forward pass for layer", layers_[i]->type_);  
    layers_[i]->Forward(layers_[i-1], passnum);    
    layers_[i]->CalcWeights(layers_[i-1], passnum);
    layers_[i]->Nonlinear(passnum);
    if (utIsInterruptPending()) {
      mexAssert(false, "Ctrl-C Detected. END");
    }
  }
  pred.attach(layers_.back()->activ_mat_);  
  //mexPrintMsg("Forward pass finished");
  /*
  if (print == 1) {
  mexPrintMsg("PRED");    
  Mat m;
  m.attach(pred);
  mexPrintMsg("s1", m.size1());    
  mexPrintMsg("s2", m.size2()); 
  mexPrintMsg("totalsum", m.sum());    
  Mat versum(1, m.size2());
  Sum(m, versum, 1);
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("versum", versum(0, i));    
  }
  Mat horsum(m.size1(), 1);
  Sum(m, horsum, 2);
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("horsum", horsum(i, 0));    
  }  
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Horizontal", m(0, i));    
  }
  for (int i = 0; i < 5; ++i) {
    mexPrintMsg("Vertical", m(i, 0));    
  }
  } */
}

void Net::Backward() {
  int passnum = 2;
  for (size_t i = layers_.size() - 1; i > first_layer_; --i) {
    //mexPrintMsg("Backward pass for layer", layers_[i]->type_);    
    if (layers_[i]->function_ != "soft" || params_.lossfun_ != "logreg") {
      // special case, final derivaties are already computed in InitDeriv
      layers_[i]->Nonlinear(passnum);      
    }
    layers_[i]->CalcWeights(layers_[i-1], passnum);
    layers_[i]->Backward(layers_[i-1]);
  }
  if (first_layer_ == 0) {
    //mexPrintMsg("Backward pass for layer", layers_[0]->type_);  
    layers_[0]->CalcWeights(NULL, passnum);
    layers_[0]->Backward(NULL);  
  }
  //mexPrintMsg("Backward pass finished");  
}

void Net::InitDeriv(const Mat &labels_batch, ftype &loss) {  
  size_t batchsize = labels_batch.size1();
  size_t classes_num = labels_batch.size2();
  Layer *lastlayer = layers_.back();
  mexAssert(batchsize == lastlayer->batchsize_, 
    "The number of objects in data and label batches is different");
  mexAssert(classes_num == lastlayer->length_, 
    "Labels in batch and last layer must have equal number of classes");  
  lossmat_.resize(batchsize, classes_num);    
  lastlayer->deriv_mat_.resize(batchsize, classes_num);    
  if (params_.lossfun_ == "logreg") {
    lossmat_ = lastlayer->activ_mat_;
    // to get the log(1) = 0 after and to avoid 0/0;
    lossmat_.CondAssign(labels_batch, false, 0, 1);    
    // to void log(0) and division by 0 if there are still some;
    lossmat_.CondAssign(lossmat_, false, 0, kEps);    
    if (lastlayer->function_ == "soft") {
      // directly compute final derivatives, so Nonlinear is not needed
      lastlayer->deriv_mat_ = lastlayer->activ_mat_;
	    lastlayer->deriv_mat_ -= labels_batch;
    } else {
      lastlayer->deriv_mat_ = labels_batch;
      (lastlayer->deriv_mat_ /= lossmat_) *= -1;      
    }
    loss = -(lossmat_.Log()).sum() / batchsize;
  } else if (params_.lossfun_ == "squared") {
    lastlayer->deriv_mat_ = lastlayer->activ_mat_;
	  lastlayer->deriv_mat_ -= labels_batch;    
    lossmat_ = lastlayer->deriv_mat_;
    loss = (lossmat_ *= lastlayer->deriv_mat_).sum() / (2 * batchsize);    
  }
  if (params_.balance_) {
    lastlayer->deriv_mat_.MultVect(classcoefs_, 1);    
  }  
  lastlayer->deriv_mat_.Validate(); 
}

void Net::InitDeriv2(ftype &loss) {

  Layer *firstlayer = layers_[first_layer_];
  size_t batchsize = firstlayer->batchsize_;
  size_t length = firstlayer->length_;
  lossmat2_.resize(batchsize, length);
  lossmat2_ = firstlayer->deriv_mat_;
  lossmat2_ *= firstlayer->deriv_mat_;
  loss = lossmat2_.sum() / (2 * batchsize);
  firstlayer->activ_mat_.attach(firstlayer->deriv_mat_);  
}

void Net::UpdateWeights(size_t epoch, bool isafter) {
  weights_.Update(params_, epoch, isafter);  
}

void Net::ReadData(const mxArray *mx_data) {
  LayerInput *firstlayer = static_cast<LayerInput*>(layers_[0]);
  std::vector<size_t> data_dim = mexGetDimensions(mx_data);
  size_t mapsize1, mapsize2;
  if (kMapsOrder == kMatlabOrder) {
    mapsize1 = data_dim[0];
    mapsize2 = data_dim[1];
  } else {
    mapsize1 = data_dim[1];
    mapsize2 = data_dim[0];
  }  
  mexAssert(mapsize1 == firstlayer->mapsize_[0] && 
            mapsize2 == firstlayer->mapsize_[1],
    "Data and the first layer must have equal sizes");  
  size_t outputmaps = 1;
  if (data_dim.size() > 2) {
    outputmaps = data_dim[2];
  }
  mexAssert(outputmaps == firstlayer->outputmaps_,
    "Data's 3rd dimension must be equal to the outputmaps on the input layer");
  size_t samples_num = 1;  
  if (data_dim.size() > 3) {
    samples_num = data_dim[3];
  }  
  ftype *data_ptr = mexGetPointer(mx_data);  
  // transposed array
  data_.attach(data_ptr, samples_num, mapsize1 * mapsize2 * outputmaps, 1, true);
  if (firstlayer->norm_ > 0) {
    MatCPU norm_data(data_.size1(), data_.size2());
    norm_data.reorder(true, false);
    norm_data = data_;
    norm_data.Normalize(firstlayer->norm_);        
    Swap(data_, norm_data);
  }
}

void Net::ReadLabels(const mxArray *mx_labels) {
  std::vector<size_t> labels_dim = mexGetDimensions(mx_labels);  
  mexAssert(labels_dim.size() == 2, "The label array must have 2 dimensions");
  size_t samples_num = labels_dim[0];
  size_t classes_num = labels_dim[1];
  mexAssert(classes_num == layers_.back()->length_,
    "Labels and last layer must have equal number of classes");  
  MatCPU labels_norm;
  mexGetMatrix(mx_labels, labels_norm); // order_ == kMatlabOrder
  if (params_.balance_) {  
    MatCPU labels_mean(1, classes_num);
    Mean(labels_norm, labels_mean, 1);
    mexAssert(!labels_mean.hasZeros(), 
      "Balancing impossible: one of the classes is not presented");
    MatCPU cpucoeffs(1, classes_num);
    cpucoeffs.assign(1);
    cpucoeffs /= labels_mean;
    classcoefs_.resize(1, classes_num);
    classcoefs_ = cpucoeffs;
    classcoefs_ /= (ftype) classes_num;
  }
  labels_.resize(samples_num, classes_num);
  labels_.reorder(true, false); // order_ == true;
  labels_ = labels_norm; 
}

void Net::InitWeights(const mxArray *mx_weights_in) {
  //mexPrintMsg("start init weights");
  bool isgen = false;
	size_t num_weights = NumWeights();  
  MatCPU weights_cpu;
  if (mx_weights_in != NULL) { // training, testing
    mexAssert(num_weights == mexGetNumel(mx_weights_in), 
      "In InitWeights the vector of weights has the wrong length!");        
    mexGetMatrix(mx_weights_in, weights_cpu);
  } else { // genweights        
    isgen = true;
    weights_cpu.resize(num_weights, 1);
  }  
  weights_mat_.resize(num_weights, 1);
  // we can attach (in CPU version), 
  // but don't want to change the initial weights
  weights_mat_ = weights_cpu;    
  weights_.Init(weights_mat_);
  size_t offset = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->InitWeights(weights_, offset, isgen);
  }  
  //mexPrintMsg("finish init weights");
}

void Net::GetWeights(mxArray *&mx_weights) const {  
  size_t num_weights = NumWeights();  
  mx_weights = mexNewMatrix(num_weights, 1);    
  Mat weights_cpu;
  weights_cpu.attach(mexGetPointer(mx_weights), num_weights, 1);
  Mat weights_mat(num_weights, 1);
  #if COMP_REGIME != 2 // CPU
    weights_mat.attach(weights_cpu);
  #endif
  size_t offset = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->GetWeights(weights_mat, offset);
  }  
  #if COMP_REGIME == 2 // GPU
    DeviceToHost(weights_mat, weights_cpu);    
  #endif    
}

void Net::GetErrors(mxArray *&mx_errors) const {  
  //mexPrintMsg("get errors");
  mx_errors = mexSetMatrix(trainerrors_);
  //mexPrintMsg("get errors end");
}

size_t Net::NumWeights() const {
  size_t num_weights = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {    
    num_weights += layers_[i]->NumWeights();
  }
  return num_weights;
}

Net::~Net() {
  size_t layers_num = layers_.size();
  for (size_t i = 0; i < layers_num; ++i) {
    delete layers_[i];
  }
  layers_.clear();  
  #if COMP_REGIME == 2 // GPU
    // remove here all GPU allocated memory manually,
    // otherwise CudaReset causes crash
    weights_.Clear();
    classcoefs_.clear(); // in fact vector
    weights_mat_.clear(); // also a vector    
    lossmat_.clear();
    lossmat2_.clear();
    MatGPU::CudaReset();    
  #endif    
}
