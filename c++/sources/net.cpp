/*
Copyright (C) 2016 Sergey Demyanov.
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

Net::Net(const mxArray *mx_params) {
  mexAssertMsg(PRECISION == 1, "In the GPU version PRECISION should be 1");
  //mexPrintMsg("Start params initialization...");
  params_.Init(mx_params);
  MatGPU::InitCuda(params_.gpu_);
  MatGPU::SetMemoryLimit(params_.memory_);
  MatGPU::InitRand(params_.seed_);
  std::srand((unsigned int) params_.seed_);
  //mexPrintMsg("Params initialization finished");
}

void Net::InitLayers(const mxArray *mx_layers) {
  //mexPrintMsg("Start layers initialization...");
  size_t layers_num = mexGetNumel(mx_layers);
  //mexAssertMsg(layers_num >= 2, "The net must contain at least 2 layers");
  layers_.resize(layers_num);
  first_trained_ = layers_num;
  //mexPrintMsg("Initializing layer of type", layer_type);
  Layer *prev_layer = NULL;
  for (size_t i = 0; i < layers_num; ++i) {
    const mxArray *mx_layer = mexGetCell(mx_layers, i);
    std::string layer_type = mexGetString(mexGetField(mx_layer, "type"));
    if (i == 0) {
      mexAssertMsg(layer_type == "input", "The first layer must be the 'input' type");
    }
    if (layer_type == "input") {
      layers_[i] = new LayerInput();
    } else if (layer_type == "full") {
      layers_[i] = new LayerFull();
    } else if (layer_type == "jitt") {
      layers_[i] = new LayerJitt();
    } else if (layer_type == "conv") {
      layers_[i] = new LayerConv();
    } else if (layer_type == "deconv") {
      layers_[i] = new LayerDeconv();
    } else if (layer_type == "pool") {
      layers_[i] = new LayerPool();
    } else {
      mexAssertMsg(false, layer_type + " - unknown layer type");
    }
    //mexPrintMsg("Initializing layer of type", layer_type);
    layers_[i]->InitGeneral(mx_layer);
    layers_[i]->Init(mx_layer, prev_layer);
    mexAssertMsg(layers_[i]->function_ != "soft" || i == layers_num - 1,
                "Softmax function may be only on the last layer");
    if (layers_[i]->NumWeights() > 0 && first_trained_ > i) {
      first_trained_ = i;
    }
    prev_layer = layers_[i];
  }
  if (params_.lossfun_ == "logreg") {
    Layer *lastlayer = layers_.back();
    if (lastlayer->function_ != "soft") {
      mexPrintMsg("WARNING: logreg loss is used with non-softmax last layer");
    }
  }
  //mexAssertMsg(layer_type == "full", "The last layer must be the type of 'f'");
  //mexAssertMsg(layers_.back()->function_ == "soft" || layers_.back()->function_ == "sigm",
  //          "The last layer function must be either 'soft' or 'sigm'");
  //mexPrintMsg("Layers initialization finished");
}

void Net::Classify(const mxArray *mx_data, const mxArray *mx_labels, mxArray *&mx_pred) {

  //mexPrintMsg("Start classification...");
  ReadData(mx_data);
  ReadLabels(mx_labels);
  size_t test_num = data_.size1();
  preds_.resize(test_num, layers_.back()->length());
  preds_.set_order(true);
  MatCPU curpreds;
  if (params_.test_epochs_ > 1) {
    curpreds.resize(test_num, layers_.back()->length());
    curpreds.set_order(true);
    preds_.assign(0);
  } else {
    curpreds.attach(preds_);
  }
  size_t numbatches = DIVUP(test_num, params_.batchsize_);
  MatGPU data_batch, labels_batch, pred_batch;
  for (size_t epoch = 0; epoch < params_.test_epochs_; ++epoch) {
    size_t offset = 0;
    for (size_t batch = 0; batch < numbatches; ++batch) {
      size_t batchsize = std::min(test_num - offset, params_.batchsize_);
      data_batch.resize(batchsize, data_.size2());
      SubSet(data_, data_batch, offset, true);
      InitActiv(data_batch);
      Forward(pred_batch, PassNum::ForwardTest, GradInd::Nowhere);
      if (params_.testshift_ > 0) {
        ftype error1;
        labels_batch.resize(batchsize, labels_.size2());
        SubSet(labels_, labels_batch, offset, true);
        InitDeriv(labels_batch, error1);
        // shift and beta cannot be positive together
        Backward(PassNum::Backward, GradInd::Nowhere);
        InitActiv3(params_.testshift_, 1); // L1-norm adversarial loss
        Forward(pred_batch, PassNum::ForwardTest, GradInd::Nowhere);
      }
      SubSet(curpreds, pred_batch, offset, false);
      offset += batchsize;
      if (params_.verbose_ == 2) {
        mexPrintInt("Test epoch", epoch + 1);
        mexPrintInt("Test batch", batch + 1);
      }
    }
    if (params_.test_epochs_ > 1) {
      preds_ += curpreds;
    }
    if (params_.verbose_ == 1) {
      mexPrintInt("Test epoch", epoch + 1);
    }
  }
  if (params_.test_epochs_ > 1) {
    preds_ /= (ftype) params_.test_epochs_;
  }
  //preds_.ReorderMaps(kInternalOrder, kExternalOrder);
  preds_.Reorder(kExternalOrder);
  Dim pred_dims = layers_.back()->dims_;
  pred_dims[0] = test_num;
  mx_pred = mexSetTensor(preds_, pred_dims);
  //mexPrintMsg("Classification finished");
}

void Net::Train(const mxArray *mx_data, const mxArray *mx_labels) {

  //mexPrintMsg("Start training...");
  ReadData(mx_data);
  ReadLabels(mx_labels);

  size_t train_num = data_.size1();
  size_t numbatches = DIVUP(train_num, params_.batchsize_);
  trainerrors_.resize(params_.epochs_, 2);
  trainerrors_.assign(0);
  MatGPU data_batch, labels_batch, pred_batch;
  for (size_t epoch = 0; epoch < params_.epochs_; ++epoch) {
    //print = 1;
    if (params_.shuffle_) {
      Shuffle(data_, labels_);
    }
    StartTimer();
    //MatGPU::StartCudaTimer();
    size_t offset = 0;
    for (size_t batch = 0; batch < numbatches; ++batch) {
      size_t batchsize = std::min(train_num - offset, params_.batchsize_);
      data_batch.resize(batchsize, data_.size2());
      labels_batch.resize(batchsize, labels_.size2());
      SubSet(data_, data_batch, offset, true);
      SubSet(labels_, labels_batch, offset, true);

      ftype error1, error2;
      InitActiv(data_batch);
      Forward(pred_batch, PassNum::Forward, GradInd::Nowhere);
      InitDeriv(labels_batch, error1);
      trainerrors_(epoch, 0) += error1;
      // shift and beta cannot be positive together
      if (params_.shift_ > 0 && params_.fast_) {
        Backward(PassNum::Backward, GradInd::Nowhere);
        InitActiv3(params_.shift_, params_.normfun_);
        Forward(pred_batch, PassNum::Forward, GradInd::Nowhere);
        InitDeriv(labels_batch, error2);
      }
      Backward(PassNum::Backward, GradInd::First);
      if (params_.shift_ > 0 && !params_.fast_) {
        InitActiv3(params_.shift_, params_.normfun_);
        Forward(pred_batch, PassNum::Forward, GradInd::Nowhere);
        InitDeriv(labels_batch, error2);
        Backward(PassNum::Backward, GradInd::Second);
      }
      if (params_.shift_ == 0 && params_.beta_ > 0) {
        InitActiv2(error2, 2);
        Forward(pred_batch, PassNum::ForwardLinear, GradInd::Second);
      }
      trainerrors_(epoch, 1) += error2;
      UpdateWeights();
      offset += batchsize;
      if (params_.verbose_ == 2) {
        mexPrintInt("Epoch", epoch + 1);
        mexPrintInt("Batch", batch + 1);
      }
    } // batch
    //MatGPU::MeasureCudaTime("totaltime");
    MeasureTime("totaltime");
    if (params_.verbose_ == 1) {
      mexPrintInt("Epoch", epoch + 1);
    }
  } // epoch
  trainerrors_ /= (ftype) numbatches;
  //mexPrintMsg("Training finished");
}

void Net::Forward(MatGPU &pred, PassNum passnum, GradInd gradind) {
  size_t batchsize = layers_[0]->activ_mat_.size1();
  Layer *prev_layer = NULL;
  for (size_t i = first_layer_; i < layers_.size(); ++i) {
    /*
    if (layers_[i]->type_ == "jitt") {
      first_layer_ = i;
    } */
    //mexPrintMsg("Forward pass for layer", layers_[i]->type_);
    if (gradind != GradInd::Nowhere && layers_[i]->lr_coef_ > 0) {
      layers_[i]->WeightGrads(prev_layer, gradind);
      // no BiasGrads on the forward pass
    }
    layers_[i]->ResizeActivMat(batchsize, passnum);
    layers_[i]->TransformForward(prev_layer, passnum);
    layers_[i]->AddBias(passnum);
    layers_[i]->Nonlinear(passnum);
    layers_[i]->DropoutForward(passnum);
    prev_layer = layers_[i];
    if (utIsInterruptPending()) {
      mexAssert(false);
    }
  }

  pred.attach(layers_.back()->activ_mat_);
  //mexPrintMsg("Forward pass finished");
}

void Net::Backward(PassNum passnum, GradInd gradind) {
  Layer *prev_layer;
  //for (int i = layers_.size() - 1; i >= first_layer_; --i) {
  for (size_t j = first_layer_; j < layers_.size(); ++j) {
    size_t i = first_layer_ + layers_.size() - 1 - j;
    //mexPrintMsg("Backward pass for layer", layers_[i]->type_);
    layers_[i]->DropoutBackward();
    if (layers_[i]->function_ != "soft" || params_.lossfun_ != "logreg") {
      // special case, final derivaties are already computed in InitDeriv
      layers_[i]->Nonlinear(passnum);
    }
    if (i > 0) {
      prev_layer = layers_[i-1];
    } else {
      prev_layer = NULL;
    }
    if (gradind != GradInd::Nowhere && layers_[i]->lr_coef_ > 0) {
      layers_[i]->WeightGrads(prev_layer, gradind);
      layers_[i]->BiasGrads(gradind);
    }/*
    if (params_.shift_ == 0 && params_.beta_ == 0) {
      if (i <= first_trained_) break;
    }*/
    if (prev_layer != NULL) {
      prev_layer->ResizeDerivMat();
      layers_[i]->TransformBackward(prev_layer);
    }
    //mexPrintMsg("der sum", layers_[i]->deriv_mat_.sum());
  }
  //mexPrintMsg("Backward pass finished");
}

void Net::InitActiv(const MatGPU &data) {
  mexAssertMsg(layers_.size() >= 1 , "The net is not initialized");
  first_layer_ = 0;
  Layer *firstlayer = layers_[first_layer_];
  firstlayer->activ_mat_.attach(data);
  firstlayer->activ_mat_.Validate();
}

void Net::InitDeriv(const MatGPU &labels_batch, ftype &loss) {
  size_t batchsize = labels_batch.size1();
  size_t classes_num = labels_batch.size2();
  Layer *lastlayer = layers_.back();
  ftype matsum = lastlayer->activ_mat_.sum();
  mexAssertMsg(!std::isnan(matsum), "Training diverged");
  mexAssertMsg(batchsize == lastlayer->dims_[0],
    "The number of objects in data and label batches is different");
  mexAssertMsg(classes_num == lastlayer->length(),
    "Labels in batch and last layer must have equal number of classes");
  lossmat_.resize(batchsize, classes_num);
  lastlayer->deriv_mat_.resize_tensor(lastlayer->dims_);
  if (params_.lossfun_ == "logreg") {
    lossmat_ = lastlayer->activ_mat_;
    // to get the log(1) = 0 after and to avoid 0/0;
    lossmat_.CondAssign(labels_batch, false, kEps, 1);
    // to avoid log(0) and division by 0 if there are still some;
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
  } else if (params_.lossfun_ == "L-norm") {
    lossmat_ = lastlayer->activ_mat_;
    lastlayer->deriv_mat_ = lastlayer->activ_mat_;
    lastlayer->deriv_mat_ -= labels_batch;
    if (params_.normfun_ == 1) {
      lastlayer->deriv_mat_.Sign();
      lossmat_ *= lastlayer->deriv_mat_; // abs
      loss = lossmat_.sum() / batchsize;
    } else if (params_.normfun_ == 2) {
      lossmat_ *= lastlayer->deriv_mat_;
      loss = lossmat_.sum() / (2 * batchsize);
    }
  }
  if (params_.balance_) {
    lastlayer->deriv_mat_.MultVect(classcoefs_, 1);
  }
  lastlayer->deriv_mat_.Validate();
}

void Net::InitActiv2(ftype &loss, int normfun) {
  Layer *firstlayer = layers_[first_layer_];
  size_t batchsize = firstlayer->dims_[0];
  size_t length = firstlayer->length();
  lossmat2_.resize(batchsize, length);
  lossmat2_ = firstlayer->deriv_mat_;
  firstlayer->activ_mat_ = firstlayer->deriv_mat_;
  if (normfun == 1) { // L1-norm
    firstlayer->activ_mat_.Sign();
    lossmat2_ *= firstlayer->activ_mat_; // abs
    loss = lossmat2_.sum() / batchsize;
  } else if (normfun == 2) { // L2-norm
    lossmat2_ *= firstlayer->activ_mat_;
    loss = lossmat2_.sum() / (2 * batchsize);
  }
}

void Net::InitActiv3(ftype coef, int normfun) {
  Layer *firstlayer = layers_[first_layer_];
  if (normfun == 1) {
    firstlayer->deriv_mat_.Sign();
  }
  firstlayer->deriv_mat_ *= coef;
  firstlayer->activ_mat_ += firstlayer->deriv_mat_;
  firstlayer->activ_mat_.Validate();
}

void Net::UpdateWeights() {
  weights_.Update(params_);
}

void Net::ReadData(const mxArray *mx_data) {
  Dim dims = mexGetTensor(mx_data, data_);
  mexAssertMsg(layers_[0]->dims_[1] == dims[1] &&
            layers_[0]->dims_[2] == dims[2] &&
            layers_[0]->dims_[3] == dims[3],
            "Data dimensions don't correspond to the input layer");
}

void Net::ReadLabels(const mxArray *mx_labels) {
  Dim dims = mexGetTensor(mx_labels, labels_);
  mexAssertMsg(layers_.back()->dims_[1] == dims[1] &&
            layers_.back()->dims_[2] == dims[2] &&
            layers_.back()->dims_[3] == dims[3],
            "Label's dimensions don't correspond to the output layer");
}

void Net::InitWeights(const mxArray *mx_weights_in) {
  //mexPrintMsg("start init weights");
  bool isgen = false;
  size_t num_weights = NumWeights();
  MatCPU weights_cpu;
  if (mx_weights_in != NULL) { // training, testing
    //mexPrintInt("num_weights", num_weights);
    //mexPrintInt("mexGetNumel", mexGetNumel(mx_weights_in));
    mexAssertMsg(num_weights == mexGetNumel(mx_weights_in),
      "The vector of weights has the wrong length!");
    mexGetMatrix(mx_weights_in, weights_cpu);
  } else { // genweights
    isgen = true;
    weights_cpu.resize(1, num_weights);
  }
  weights_.Init(weights_cpu);
  size_t offset = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->InitWeights(weights_, offset, isgen);
  }
  //mexPrintMsg("finish init weights");
}

void Net::GetWeights(mxArray *&mx_weights) const {
  size_t num_weights = NumWeights();
  mx_weights = mexNewMatrix(1, num_weights);
  MatCPU weights_cpu;
  weights_cpu.attach(mexGetPointer(mx_weights), 1, num_weights);
  size_t offset = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->RestoreOrder();
  }
  DeviceToHost(weights_.get(), weights_cpu);
}

void Net::GetErrors(mxArray *&mx_errors) const {
  mx_errors = mexSetMatrix(trainerrors_);
}

size_t Net::NumWeights() const {
  size_t num_weights = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    num_weights += layers_[i]->NumWeights();
    //mexPrintInt("i", num_weights);
  }
  return num_weights;
}

Net::~Net() {
  for (size_t i = 0; i < layers_.size(); ++i) {
    delete layers_[i];
  }
  layers_.clear();
  // remove here all GPU allocated memory manually,
  // otherwise CudaReset causes crash
  weights_.Clear();
  classcoefs_.clear(); // in fact vector
  lossmat_.clear();
  lossmat2_.clear();
  MatGPU::CudaReset();
}
